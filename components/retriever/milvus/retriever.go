package milvus

import (
	"context"
	"fmt"
	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/retriever"
	"github.com/cloudwego/eino/schema"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

const (
	defaultTopK        = 100
	defaultPartition   = "default"
	defaultDenseWeight = 0.5
)

type RetrieverConfig struct {
	Address           string `json:"address"`            // Remote address, "localhost:19530".
	Username          string `json:"username"`           // Username for auth.
	Password          string `json:"password"`           // Password for auth.
	DBName            string `json:"dbname"`             // DBName for this client.
	ConnectionTimeout int64  `json:"connection_timeout"` // second

	Collection string `json:"collection"`
	Index      string `json:"index"`

	EmbeddingConfig EmbeddingConfig `json:"embedding_config"`

	// Partition 子索引划分字段, 索引中未配置时至空即可
	Partition string `json:"partition"`
	// TopK will be set with 100 if zero
	TopK           *int     `json:"top_k,omitempty"`
	ScoreThreshold *float64 `json:"score_threshold,omitempty"`
	// FilterDSL 标量过滤 filter 表达式 https://www.volcengine.com/docs/84313/1254609
	FilterDSL map[string]any `json:"filter_dsl,omitempty"`
}

type EmbeddingConfig struct {
	// UseBuiltin 是否使用 VikingDB 内置向量化方法 (embedding v2)
	// true 时需要配置 ModelName 和 UseSparse, false 时需要配置 Embedding
	// see: https://www.volcengine.com/docs/84313/1254568
	UseBuiltin bool `json:"use_builtin"`

	// ModelName 指定模型名称
	ModelName string `json:"model_name"`
	// UseSparse 是否返回稀疏向量
	// 支持提取稀疏向量的模型设置为 true 返回稠密+稀疏向量，设置为 false 仅返回稠密向量
	// 不支持稀疏向量的模型设置为 true 会报错
	UseSparse bool `json:"use_sparse"`
	// DenseWeight 对于标量过滤检索，dense_weight 用于控制稠密向量在检索中的权重。范围为[0.2，1], 仅在检索的索引为混合索引时有效
	// 默认值为 0.5
	DenseWeight float64 `json:"dense_weight"`

	// Embedding 使用自行指定的 embedding 替换 VikingDB 内置向量化方法
	Embedding embedding.Embedder
}

type Retriever struct {
	config *RetrieverConfig
	client client.Client
}

func NewRetriever(ctx context.Context, config *RetrieverConfig) (*Retriever, error) {
	if config.EmbeddingConfig.UseBuiltin && config.EmbeddingConfig.Embedding != nil {
		return nil, fmt.Errorf("[VikingDBRetriever] no need to provide Embedding when UseBuiltin embedding is true")
	} else if !config.EmbeddingConfig.UseBuiltin && config.EmbeddingConfig.Embedding == nil {
		return nil, fmt.Errorf("[VikingDBRetriever] need provide Embedding when UseBuiltin embedding is false")
	}

	mc, err := client.NewClient(ctx, client.Config{
		Address:  config.Address,
		Username: config.Username,
		Password: config.Password,
		DBName:   config.DBName,
	})
	if err != nil {
		return nil, err
	}

	if len(config.Partition) == 0 {
		config.Partition = defaultPartition
	}

	r := &Retriever{
		config: config,
		client: mc,
	}

	return r, nil
}

func (r *Retriever) Retrieve(ctx context.Context, query string, opts ...retriever.Option) (docs []*schema.Document, err error) {
	defer func() {
		if err != nil {
			ctx = callbacks.OnError(ctx, err)
		}
	}()

	options := retriever.GetCommonOptions(&retriever.Options{
		Index:          &r.config.Index,
		SubIndex:       &r.config.Partition,
		TopK:           r.config.TopK,
		ScoreThreshold: r.config.ScoreThreshold,
		Embedding:      r.config.EmbeddingConfig.Embedding,
		DSLInfo:        r.config.FilterDSL,
	}, opts...)

	var (
		dense []float32
	)

	ctx = callbacks.OnStart(ctx, &retriever.CallbackInput{
		Query:          query,
		TopK:           dereferenceOrZero(options.TopK),
		Filter:         tryMarshalJsonString(options.DSLInfo),
		ScoreThreshold: options.ScoreThreshold,
	})

	dense, err = r.customEmbedding(ctx, query, options)

	if err != nil {
		return nil, err
	}
	vector := entity.FloatVector(dense)
	sp, _ := entity.NewIndexFlatSearchParam()
	result, err := r.client.Search(ctx, r.config.Collection, []string{}, "", []string{defaultReturnFieldID, defaultReturnFieldContent},
		[]entity.Vector{vector}, defaultFieldVector, entity.L2, *r.config.TopK, sp)
	if err != nil {
		return nil, err
	}

	docs = make([]*schema.Document, 0, len(result))
	for _, data := range result {
		doc, err := r.data2Document(data)
		if err != nil {
			return nil, err
		}

		docs = append(docs, doc.WithDSLInfo(options.DSLInfo))
	}

	ctx = callbacks.OnEnd(ctx, &retriever.CallbackOutput{Docs: docs})

	return docs, nil
}

func (r *Retriever) customEmbedding(ctx context.Context, query string, options *retriever.Options) (vector []float32, err error) {
	emb := options.Embedding
	tempVectors, err := emb.EmbedStrings(r.makeEmbeddingCtx(ctx, emb), []string{query})
	if err != nil {
		return nil, err
	}

	if len(tempVectors) != 1 { // unexpected
		return nil, fmt.Errorf("[customEmbedding] invalid return length of vector, got=%d, expected=1", len(tempVectors))
	}

	firstVector := tempVectors[0]
	vector = make([]float32, len(firstVector))
	for idx := range firstVector {
		vector[idx] = float32(firstVector[idx])
	}

	return
}

func (r *Retriever) makeEmbeddingCtx(ctx context.Context, emb embedding.Embedder) context.Context {
	runInfo := &callbacks.RunInfo{
		Component: components.ComponentOfEmbedding,
	}

	if embType, ok := components.GetType(emb); ok {
		runInfo.Type = embType
	}

	runInfo.Name = runInfo.Type + string(runInfo.Component)

	return callbacks.ReuseHandlers(ctx, runInfo)
}

func (r *Retriever) data2Document(data client.SearchResult) (*schema.Document, error) {
	var idColumn *entity.ColumnVarChar
	var contentColumn *entity.ColumnVarChar
	for _, field := range data.Fields {
		if field.Name() == defaultReturnFieldID {
			c, ok := field.(*entity.ColumnVarChar)
			if ok {
				idColumn = c
			}
		} else if field.Name() == defaultReturnFieldContent {
			c, ok := field.(*entity.ColumnVarChar)
			if ok {
				contentColumn = c
			}
		}
	}
	if idColumn == nil || contentColumn == nil || data.ResultCount == 0 {
		return nil, fmt.Errorf("result field not math")
	}

	id, err := idColumn.ValueByIdx(0)
	if err != nil {
		return nil, err
	}
	content, err := contentColumn.ValueByIdx(0)
	if err != nil {
		return nil, err
	}
	doc := &schema.Document{
		ID:       id,
		Content:  content,
		MetaData: map[string]any{},
	}

	doc.WithScore(float64(data.Scores[0]))

	return doc, nil
}

func (r *Retriever) GetType() string {
	return typ
}

func (r *Retriever) IsCallbacksEnabled() bool {
	return true
}
