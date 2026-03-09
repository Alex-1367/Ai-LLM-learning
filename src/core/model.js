// src/core/model.js
import { Matrix, Activations } from './matrix.js';
import EmbeddingLayer from './embeddings.js';
import MultiHeadAttention from './attention.js';

class TransformerBlock {
  constructor(embeddingDim, numHeads) {
    this.attention = new MultiHeadAttention(embeddingDim, numHeads);
    
    // Feed-forward network (simplified)
    this.ffn1 = new Matrix(embeddingDim, embeddingDim * 4);
    this.ffn2 = new Matrix(embeddingDim * 4, embeddingDim);
    
    // Layer normalization parameters
    this.ln1 = { gamma: new Matrix(1, embeddingDim).map(() => 1), beta: new Matrix(1, embeddingDim).map(() => 0) };
    this.ln2 = { gamma: new Matrix(1, embeddingDim).map(() => 1), beta: new Matrix(1, embeddingDim).map(() => 0) };
  }

  forward(x) {
    // Pre-norm architecture (modern LLMs use this)
    
    // 1. Self-attention with residual connection
    const normalized1 = this.layerNorm(x, this.ln1);
    const attnOutput = this.attention.forward(normalized1);
    const x1 = x.add(attnOutput);
    
    // 2. Feed-forward with residual connection
    const normalized2 = this.layerNorm(x1, this.ln2);
    const ffOutput = this.feedForward(normalized2);
    const x2 = x1.add(ffOutput);
    
    return x2;
  }

  feedForward(x) {
    // FFN(x) = GELU(xW1)W2
    const hidden = x.multiply(this.ffn1).map(Activations.gelu);
    return hidden.multiply(this.ffn2);
  }

  layerNorm(x, params) {
    const result = new Matrix(x.rows, x.cols);
    
    for (let i = 0; i < x.rows; i++) {
      const mean = x.data[i].reduce((a, b) => a + b, 0) / x.cols;
      const variance = x.data[i].reduce((a, b) => a + Math.pow(b - mean, 2), 0) / x.cols;
      const std = Math.sqrt(variance + 1e-5);
      
      for (let j = 0; j < x.cols; j++) {
        result.data[i][j] = params.gamma.data[0][j] * (x.data[i][j] - mean) / std + params.beta.data[0][j];
      }
    }
    
    return result;
  }
}


export default class TinyLLM {
  constructor(config, tokenizer) {
    this.tokenizer = tokenizer;
    this.vocabSize = config.vocabSize || tokenizer.vocabSize;
    this.embeddingDim = config.embeddingDim || 64;
    this.numLayers = config.numLayers || 2;
    this.numHeads = config.numHeads || 4;
    this.maxLength = config.maxLength || 64;
    
    console.log('\n🏗️  Building TinyLLM:');
    console.log(`   Vocabulary: ${this.vocabSize} words`);
    console.log(`   Embedding dimensions: ${this.embeddingDim}`);
    console.log(`   Transformer layers: ${this.numLayers}`);
    console.log(`   Attention heads: ${this.numHeads}`);
    
    // Core components
    this.embedding = new EmbeddingLayer(this.vocabSize, this.embeddingDim);
    
    // Positional encoding (so model knows word order)
    this.positionalEncoding = this.createPositionalEncoding();
    
    // Stack of transformer blocks
    this.layers = [];
    for (let i = 0; i < this.numLayers; i++) {
      this.layers.push(new TransformerBlock(this.embeddingDim, this.numHeads));
    }
    
    // Final output projection (back to vocabulary)
    this.outputProjection = new Matrix(this.embeddingDim, this.vocabSize);
    
    console.log('✅ Model architecture created\n');
  }

  createPositionalEncoding() {
    // Sinusoidal positional encodings (from "Attention Is All You Need")
    const pe = new Matrix(this.maxLength, this.embeddingDim);
    
    for (let pos = 0; pos < this.maxLength; pos++) {
      for (let i = 0; i < this.embeddingDim; i++) {
        if (i % 2 === 0) {
          pe.data[pos][i] = Math.sin(pos / Math.pow(10000, i / this.embeddingDim));
        } else {
          pe.data[pos][i] = Math.cos(pos / Math.pow(10000, (i - 1) / this.embeddingDim));
        }
      }
    }
    
    return pe;
  }

  forward(tokens) {
    // tokens: array of token indices [seqLength]
    const seqLength = tokens.length;
    
    // 1. Get word embeddings
    const wordEmbeddings = this.embedding.forward(tokens);
    
    // 2. Add positional encodings
    for (let pos = 0; pos < seqLength; pos++) {
      for (let dim = 0; dim < this.embeddingDim; dim++) {
        wordEmbeddings.data[pos][dim] += this.positionalEncoding.data[pos][dim];
      }
    }
    
    // 3. Pass through transformer layers
    let x = wordEmbeddings;
    for (let layer of this.layers) {
      x = layer.forward(x);
    }
    
    // 4. Project to vocabulary
    const logits = x.multiply(this.outputProjection);
    
    return logits; // [seqLength, vocabSize]
  }

  // Generate text (like Ollama's generate)
  generate(prompt, maxNewTokens = 20, temperature = 0.8) {
    console.log(`\n🤔 Generating from: "${prompt}"`);
    
    let tokens = this.tokenizer.encode(prompt);
    const originalLength = tokens.length;
    
    for (let i = 0; i < maxNewTokens; i++) {
      // Forward pass
      const logits = this.forward(tokens.slice(-this.maxLength));
      
      // Get logits for last position
      const lastLogits = logits.data[logits.rows - 1];
      
      // Apply temperature and sample
      const probs = lastLogits.map(x => Math.exp(x / temperature));
      const sum = probs.reduce((a, b) => a + b, 0);
      const normalized = probs.map(x => x / sum);
      
      // Sample from distribution (adds randomness/creativity)
      const nextToken = this.sampleFromDistribution(normalized);
      tokens.push(nextToken);
      
      // Stop if we hit EOS
      if (nextToken === this.tokenizer.specialTokens['<EOS>']) break;
    }
    
    const generated = this.tokenizer.decode(tokens.slice(originalLength));
    console.log(`✨ Generated: "${prompt}${generated ? ' ' + generated : ''}"`);
    
    return {
      full: this.tokenizer.decode(tokens),
      new: generated
    };
  }

  sampleFromDistribution(probs) {
    const r = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) return i;
    }
    
    return probs.length - 1;
  }

  // Calculate loss (how wrong the model is)
  computeLoss(predictions, targets) {
    // predictions: [seqLength, vocabSize]
    // targets: [seqLength] (true next tokens)
    
    let totalLoss = 0;
    let correct = 0;
    let total = 0;
    
    for (let i = 0; i < predictions.rows - 1; i++) {
      // We predict next token from current position
      const trueToken = targets[i + 1];
      const predProbs = predictions.data[i];
      
      // Cross-entropy loss: -log(p[true_token])
      totalLoss += -Math.log(predProbs[trueToken] + 1e-10);
      
      // Accuracy
      const predictedToken = predProbs.indexOf(Math.max(...predProbs));
      if (predictedToken === trueToken) correct++;
      total++;
    }
    
    return {
      loss: totalLoss / total,
      accuracy: correct / total
    };
  }

  // Visualize what the model learned
  visualizeWordVectors(words) {
    console.log('\n📊 Word Vector Analysis:');
    for (const word of words) {
      this.embedding.visualizeWord(word, this.tokenizer);
      
      const similar = this.embedding.findSimilarWords(word, this.tokenizer, 3);
      console.log(`Similar words: ${similar.map(s => `${s.word} (${s.similarity})`).join(', ')}`);
      console.log('---');
    }
  }
}

