package main

import (
	"context"
	"fmt"
	"github.com/cloudwego/eino-ext/components/indexer/milvus"
	"github.com/cloudwego/eino/callbacks"
	"github.com/cloudwego/eino/components/embedding"
	"github.com/cloudwego/eino/components/indexer"
	"github.com/cloudwego/eino/compose"
	"github.com/cloudwego/eino/schema"
	callbacksHelper "github.com/cloudwego/eino/utils/callbacks"
	"log"
)

func main() {
	ctx := context.Background()
	collectionName := "eino_test"

	/*
	 * 下面示例中提前构建了一个名为 eino_test 的数据集 (collection)，字段配置为:
	 * 字段名称			字段类型			向量维度
	 * ID				string
	 * vector			vector			1024
	 * content			string
	 *
	 * component 使用时注意:
	 * 1. ID / vector / content 的字段名称与类型与上方配置一致
	 * 2. vector 向量维度需要与 ModelName 对应的模型所输出的向量维度一致
	 * 3. 部分模型不输出稀疏向量，此时 UseSparse 需要设置为 false，collection 可以不设置 sparse_vector 字段
	 */

	cfg := &milvus.IndexerConfig{
		Address:    "localhost:19530",
		Collection: collectionName,
		EmbeddingConfig: milvus.EmbeddingConfig{
			Embedding: &mockEmbedding{},
		},
		AddBatchSize: 10,
	}

	milvusIndexer, err := milvus.NewIndexer(ctx, cfg)
	if err != nil {
		fmt.Printf("NewIndexer failed, %v\n", err)
		return
	}

	doc := &schema.Document{
		ID:      "mock_id_1",
		Content: "A ReAct prompt consists of few-shot task-solving trajectories, with human-written text reasoning traces and actions, as well as environment observations in response to actions",
	}

	docs := []*schema.Document{doc}

	log.Printf("===== call Indexer directly =====")

	resp, err := milvusIndexer.Store(ctx, docs)
	if err != nil {
		fmt.Printf("Store failed, %v\n", err)
		return
	}

	fmt.Printf("milvus store success, docs=%v, resp ids=%v\n", docs, resp)

	log.Printf("===== call Indexer in chain =====")

	// 创建 callback handler
	handlerHelper := &callbacksHelper.IndexerCallbackHandler{
		OnStart: func(ctx context.Context, info *callbacks.RunInfo, input *indexer.CallbackInput) context.Context {
			log.Printf("input access, len: %v, content: %s\n", len(input.Docs), input.Docs[0].Content)
			return ctx
		},
		OnEnd: func(ctx context.Context, info *callbacks.RunInfo, output *indexer.CallbackOutput) context.Context {
			log.Printf("output finished, len: %v, ids=%v\n", len(output.IDs), output.IDs)
			return ctx
		},
		// OnError
	}

	// 使用 callback handler
	handler := callbacksHelper.NewHandlerHelper().
		Indexer(handlerHelper).
		Handler()

	chain := compose.NewChain[[]*schema.Document, []string]()
	chain.AppendIndexer(milvusIndexer)

	// 在运行时使用
	run, err := chain.Compile(ctx)
	if err != nil {
		log.Fatalf("chain.Compile failed, err=%v", err)
	}

	outIDs, err := run.Invoke(ctx, docs, compose.WithCallbacks(handler))
	if err != nil {
		log.Fatalf("run.Invoke failed, err=%v", err)
	}
	fmt.Printf("milvus store success, docs=%v, resp ids=%v\n", docs, outIDs)
}

type mockEmbedding struct{}

func (m mockEmbedding) EmbedStrings(ctx context.Context, texts []string, opts ...embedding.Option) ([][]float64, error) {
	slice := make([]float64, 1024)
	for i := range slice {
		slice[i] = 1.1
	}

	return [][]float64{slice}, nil
}
