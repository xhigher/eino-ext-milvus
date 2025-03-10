package milvus

import (
	"context"
	"fmt"
	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/schema"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"strconv"
)

type IndexerConfig struct {
	Address    string `json:"address"`  // Remote address, "localhost:19530".
	Username   string `json:"username"` // Username for auth.
	Password   string `json:"password"` // Password for auth.
	DBName     string `json:"dbname"`   // DBName for this client.
	Collection string `json:"collection"`

	EmbeddingConfig EmbeddingConfig `json:"embedding_config"`

	AddBatchSize int `json:"add_batch_size"`
}

type EmbeddingConfig struct {
	UseBuiltin bool `json:"use_builtin"`
	// ModelName 指定模型名称
	ModelName string `json:"model_name"`
	// UseSparse 是否返回稀疏向量
	// 支持提取稀疏向量的模型设置为 true 返回稠密+稀疏向量，设置为 false 仅返回稠密向量
	// 不支持稀疏向量的模型设置为 true 会报错
	UseSparse bool `json:"use_sparse"`

	VectorDim int `json:"vector_dim"`
	IdMaxLen  int `json:"id_max_len"`

	// Embedding when UseBuiltin is false
	// If Embedding from here or from indexer.Option is provided, it will take precedence over built-in vectorization methods
	Embedding embedding.Embedder
}

type Indexer struct {
	config *IndexerConfig
	client client.Client
}

type Columns struct {
	ID           *entity.ColumnVarChar
	Content      *entity.ColumnVarChar
	Vector       *entity.ColumnFloatVector
	SparseVector *entity.ColumnFloatVector
}

func NewIndexer(ctx context.Context, config *IndexerConfig) (*Indexer, error) {
	if config.EmbeddingConfig.UseBuiltin && config.EmbeddingConfig.Embedding != nil {
		return nil, fmt.Errorf("[MilvusDBIndexer] no need to provide Embedding when UseBuiltin embedding is true")
	} else if !config.EmbeddingConfig.UseBuiltin && config.EmbeddingConfig.Embedding == nil {
		return nil, fmt.Errorf("[MilvusDBIndexer] need provide Embedding when UseBuiltin embedding is false")
	}

	if config.AddBatchSize == 0 {
		config.AddBatchSize = defaultAddBatchSize
	}
	if config.EmbeddingConfig.VectorDim == 0 {
		config.EmbeddingConfig.VectorDim = defaultVectorDim
	}
	if config.EmbeddingConfig.IdMaxLen == 0 {
		config.EmbeddingConfig.IdMaxLen = defaultIdMaxLen
	}

	mc, err := client.NewClient(context.Background(), client.Config{
		Address:  config.Address,
		Username: config.Username,
		Password: config.Password,
		DBName:   config.DBName,
	})
	if err != nil {
		return nil, err
	}

	has, err := mc.HasCollection(ctx, config.Collection)
	if err != nil {
		return nil, err
	}
	if !has {
		entitySchema := &entity.Schema{
			CollectionName: config.Collection,
			Description:    "this is the example collection for inser and search",
			AutoID:         false,
			Fields: []*entity.Field{
				{
					Name:       defaultFieldID,
					DataType:   entity.FieldTypeVarChar,
					PrimaryKey: true,
					TypeParams: map[string]string{
						entity.TypeParamMaxLength: strconv.Itoa(config.EmbeddingConfig.IdMaxLen),
					},
				},
				{
					Name:     defaultFieldContent,
					DataType: entity.FieldTypeVarChar,
					TypeParams: map[string]string{
						entity.TypeParamMaxLength: strconv.Itoa(65535),
					},
				},
				{
					Name:     defaultFieldVector,
					DataType: entity.FieldTypeFloatVector,
					TypeParams: map[string]string{
						entity.TypeParamDim: strconv.Itoa(config.EmbeddingConfig.VectorDim),
					},
				},
			},
		}

		err = mc.CreateCollection(ctx, entitySchema, entity.DefaultShardNumber) // only 1 shard
		if err != nil {
			return nil, err
		}
	} else {
		//err = mc.DropCollection(ctx, config.Collection)
		//if err != nil {
		//	return nil, err
		//}
	}

	i := &Indexer{
		config: config,
		client: mc,
	}

	if config.EmbeddingConfig.UseBuiltin {
		i.embModel = &models.NewTextEmbeddingFunction
	}

	return i, nil
}

func (i *Indexer) Store(ctx context.Context, docs []*schema.Document, opts ...indexer.Option) (ids []string, err error) {
	defer func() {
		if err != nil {
			ctx = callbacks.OnError(ctx, err)
		}
	}()

	options := indexer.GetCommonOptions(&indexer.Options{
		Embedding: i.config.EmbeddingConfig.Embedding,
	}, opts...)

	ctx = callbacks.OnStart(ctx, &indexer.CallbackInput{Docs: docs})

	ids = make([]string, 0, len(docs))
	for _, sub := range chunk(docs, i.config.AddBatchSize) {
		columns, err := i.convertDocuments(ctx, sub, options)
		if err != nil {
			return nil, fmt.Errorf("convertDocuments failed: %w", err)
		}

		if _, err = i.client.Upsert(ctx, i.config.Collection, "", columns.ID, columns.Content, columns.Vector); err != nil {
			return nil, fmt.Errorf("Upsert failed: %v", err)
		}

		ids = append(ids, iter(sub, func(t *schema.Document) string { return t.ID })...)
	}

	ctx = callbacks.OnEnd(ctx, &indexer.CallbackOutput{IDs: ids})

	return ids, nil
}

func (i *Indexer) convertDocuments(ctx context.Context, docs []*schema.Document, options *indexer.Options) (columns *Columns, err error) {
	var (
		vectors [][]float32
	)

	queries := iter(docs, func(doc *schema.Document) string {
		return doc.Content
	})

	vectors, err = i.customEmbedding(ctx, queries, options)
	if err != nil {
		return
	}

	size := len(docs)
	ids := make([]string, size)
	contents := make([]string, size)
	for idx := range docs {
		doc := docs[idx]
		ids[idx] = doc.ID
		contents[idx] = doc.Content
	}

	columns = &Columns{
		ID:      entity.NewColumnVarChar(defaultFieldID, ids),
		Content: entity.NewColumnVarChar(defaultFieldContent, contents),
		Vector:  entity.NewColumnFloatVector(defaultFieldVector, i.getVectorDim(), vectors),
	}

	return
}

func (i *Indexer) customEmbedding(ctx context.Context, queries []string, options *indexer.Options) (vectors [][]float32, err error) {
	emb := options.Embedding
	tempVectors, err := emb.EmbedStrings(i.makeEmbeddingCtx(ctx, emb), queries)
	if err != nil {
		return
	}

	if len(tempVectors) != len(queries) {
		err = fmt.Errorf("[customEmbedding] invalid return length of vector, got=%d, expected=%d", len(vectors), len(queries))
		return
	}
	vectors = make([][]float32, len(tempVectors))
	for idx, values := range tempVectors {
		vectors[idx] = make([]float32, len(values))
		for j, value := range values {
			vectors[idx][j] = float32(value)
		}
	}

	return
}

func (i *Indexer) makeEmbeddingCtx(ctx context.Context, emb embedding.Embedder) context.Context {
	runInfo := &callbacks.RunInfo{
		Component: components.ComponentOfEmbedding,
	}

	if embType, ok := components.GetType(emb); ok {
		runInfo.Type = embType
	}

	runInfo.Name = runInfo.Type + string(runInfo.Component)

	return callbacks.ReuseHandlers(ctx, runInfo)
}

func (i *Indexer) GetType() string {
	return typ
}

func (i *Indexer) IsCallbacksEnabled() bool {
	return true
}

func (i *Indexer) getVectorDim() int {
	return i.config.EmbeddingConfig.VectorDim
}
