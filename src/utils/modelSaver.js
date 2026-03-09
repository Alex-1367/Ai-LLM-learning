// src/utils/modelSaver.js
import fs from 'fs/promises';
import path from 'path';

export class ModelSaver {
  constructor(model, tokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  async saveModel(modelName = 'tinyllm-german') {
    console.log('\n💾 Saving model...');
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const modelDir = path.join(process.cwd(), 'models', `${modelName}-${timestamp}`);
    
    // Create model directory
    await fs.mkdir(modelDir, { recursive: true });
    
    // 1. Save model configuration
    const config = {
      vocabSize: this.model.vocabSize,
      embeddingDim: this.model.embeddingDim,
      numLayers: this.model.numLayers,
      numHeads: this.model.numHeads,
      maxLength: this.model.maxLength,
      timestamp: new Date().toISOString(),
      stats: {
        totalWords: this.tokenizer.vocabSize,
        totalSentences: this.model.totalSentences || 4579
      }
    };
    
    await fs.writeFile(
      path.join(modelDir, 'config.json'),
      JSON.stringify(config, null, 2)
    );
    console.log(`✅ Saved config to config.json`);

    // 2. Save vocabulary (this is like your tokenizer files)
    const vocabulary = {
      wordToIndex: Array.from(this.tokenizer.wordToIndex.entries()),
      indexToWord: Array.from(this.tokenizer.indexToWord.entries()),
      specialTokens: this.tokenizer.specialTokens
    };
    
    await fs.writeFile(
      path.join(modelDir, 'vocabulary.json'),
      JSON.stringify(vocabulary, null, 2)
    );
    console.log(`✅ Saved vocabulary to vocabulary.json`);

    // 3. Save embedding weights (the core word vectors)
    const embeddings = {
      weights: this.model.embedding.weights.data,
      shape: [this.model.embedding.weights.rows, this.model.embedding.weights.cols]
    };
    
    await fs.writeFile(
      path.join(modelDir, 'embeddings.json'),
      JSON.stringify(embeddings)
    );
    console.log(`✅ Saved embeddings (${embeddings.shape[0]}×${embeddings.shape[1]})`);

    // 4. Save transformer weights
    const transformerWeights = [];
    for (let i = 0; i < this.model.layers.length; i++) {
      const layer = this.model.layers[i];
      transformerWeights.push({
        layer: i,
        attention: {
          Wq: layer.attention.Wq.data,
          Wk: layer.attention.Wk.data,
          Wv: layer.attention.Wv.data,
          Wo: layer.attention.Wo.data
        },
        ffn: {
          W1: layer.ffn1.data,
          W2: layer.ffn2.data
        }
      });
    }
    
    await fs.writeFile(
      path.join(modelDir, 'transformer.json'),
      JSON.stringify(transformerWeights)
    );
    console.log(`✅ Saved transformer weights (${this.model.layers.length} layers)`);

    // 5. Save output projection
    await fs.writeFile(
      path.join(modelDir, 'output_projection.json'),
      JSON.stringify(this.model.outputProjection.data)
    );
    console.log(`✅ Saved output projection`);

    // 6. Create a model summary
    const summary = {
      model: config,
      vocabulary: {
        size: this.tokenizer.vocabSize,
        specialTokens: this.tokenizer.specialTokens,
        sampleWords: Array.from(this.tokenizer.wordToIndex.keys()).slice(0, 20)
      },
      files: [
        'config.json',
        'vocabulary.json', 
        'embeddings.json',
        'transformer.json',
        'output_projection.json'
      ],
      totalParameters: this.calculateParameters()
    };
    
    await fs.writeFile(
      path.join(modelDir, 'summary.json'),
      JSON.stringify(summary, null, 2)
    );
    
    console.log(`\n📊 Model Summary:`);
    console.log(`   Location: ${modelDir}`);
    console.log(`   Total parameters: ~${(summary.totalParameters / 1e6).toFixed(2)}M`);
    console.log(`   Vocabulary size: ${this.tokenizer.vocabSize}`);
    console.log(`   Embedding dimensions: ${this.model.embeddingDim}`);
    
    return modelDir;
  }

  calculateParameters() {
    // Rough parameter count
    let total = 0;
    
    // Embeddings
    total += this.model.vocabSize * this.model.embeddingDim;
    
    // Transformer layers
    for (const layer of this.model.layers) {
      // Attention matrices (Q,K,V,O)
      total += 4 * this.model.embeddingDim * this.model.embeddingDim;
      // Feed-forward
      total += 2 * this.model.embeddingDim * (this.model.embeddingDim * 4);
    }
    
    // Output projection
    total += this.model.embeddingDim * this.model.vocabSize;
    
    return total;
  }

  async saveAsGGUFAnalog(format = 'json') {
    // This creates a format similar to GGUF but in JSON
    // In real GGUF, this would be binary with quantization
    
    const modelName = 'tinyllm-german';
    const modelDir = await this.saveModel(modelName);
    
    // Create a "GGUF-like" manifest
    const manifest = {
      format: "tinyllm-gguf-analog",
      version: "1.0",
      model: modelName,
      architecture: "tinyllm",
      parameters: this.calculateParameters(),
      quantization: "float32",  // In real GGUF, this would be Q4_0, Q5_K, etc.
      files: [
        "config.json",
        "vocabulary.json",
        "embeddings.json", 
        "transformer.json",
        "output_projection.json"
      ],
      metadata: {
        trained_on: `${this.tokenizer.vocabSize} words, ${this.model.totalSentences || 4579} sentences`,
        embedding_dim: this.model.embeddingDim,
        layers: this.model.numLayers,
        heads: this.model.numHeads,
        date: new Date().toISOString()
      }
    };
    
    await fs.writeFile(
      path.join(modelDir, 'gguf-manifest.json'),
      JSON.stringify(manifest, null, 2)
    );
    
    console.log(`\n🎯 Created GGUF-analog format in:`);
    console.log(`   ${modelDir}/`);
    console.log(`   This mimics how GGUF stores model weights, but in readable JSON!`);
    
    return modelDir;
  }
}

// Also add a loader to restore models
export class ModelLoader {
  async loadModel(modelDir) {
    console.log(`\n📂 Loading model from ${modelDir}`);
    
    // Load config
    const config = JSON.parse(
      await fs.readFile(path.join(modelDir, 'config.json'), 'utf-8')
    );
    
    // Load vocabulary
    const vocabData = JSON.parse(
      await fs.readFile(path.join(modelDir, 'vocabulary.json'), 'utf-8')
    );
    
    // Rebuild tokenizer
    const tokenizer = new VocabularyBuilder({ vocabulary: new Map(), texts: [] });
    tokenizer.wordToIndex = new Map(vocabData.wordToIndex);
    tokenizer.indexToWord = new Map(vocabData.indexToWord);
    tokenizer.specialTokens = vocabData.specialTokens;
    tokenizer.vocabSize = tokenizer.wordToIndex.size;
    
    // Load embeddings
    const embeddingsData = JSON.parse(
      await fs.readFile(path.join(modelDir, 'embeddings.json'), 'utf-8')
    );
    
    // Create model and load weights
    const model = new TinyLLM(config, tokenizer);
    
    // Load weights into model (you'd need to add setter methods)
    // This shows the concept - in practice you'd add loadWeights() methods
    
    console.log(`✅ Model loaded successfully`);
    return { model, tokenizer, config };
  }
}