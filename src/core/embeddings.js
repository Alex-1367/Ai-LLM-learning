// src/core/embeddings.js
import { Matrix } from './matrix.js';

export default class EmbeddingLayer {
  constructor(vocabSize, embeddingDim) {
    this.vocabSize = vocabSize;
    this.embeddingDim = embeddingDim;
    
    // THIS IS THE FAMOUS EMBEDDING MATRIX
    // Each word gets a vector of numbers that represent its meaning
    console.log(`🎯 Creating embedding matrix: ${vocabSize} words × ${embeddingDim} dimensions`);
    
    // Initialize with small random values (this is what gets trained)
    this.weights = new Matrix(vocabSize, embeddingDim);
    
    // For tracking gradients during training
    this.gradients = new Matrix(vocabSize, embeddingDim);
    
    // Cache for backprop
    this.lastInputIndices = null;
    this.lastOutput = null;
  }

  // Forward pass: convert token indices to vectors
  forward(inputIndices) {
    // inputIndices: array of token IDs [seqLength]
    this.lastInputIndices = inputIndices;
    
    // Create output matrix: [seqLength, embeddingDim]
    const output = new Matrix(inputIndices.length, this.embeddingDim);
    
    // For each position, look up the corresponding word vector
    for (let pos = 0; pos < inputIndices.length; pos++) {
      const tokenIdx = inputIndices[pos];
      
      // Copy the embedding for this token
      for (let dim = 0; dim < this.embeddingDim; dim++) {
        output.data[pos][dim] = this.weights.data[tokenIdx][dim];
      }
    }
    
    this.lastOutput = output;
    return output;
  }

  // Backward pass: update embeddings based on error
  backward(gradient) {
    // gradient: error signal for each position [seqLength, embeddingDim]
    
    // Reset gradients
    this.gradients = new Matrix(this.vocabSize, this.embeddingDim);
    
    // For each position, add the gradient to the corresponding word's vector
    for (let pos = 0; pos < this.lastInputIndices.length; pos++) {
      const tokenIdx = this.lastInputIndices[pos];
      
      for (let dim = 0; dim < this.embeddingDim; dim++) {
        this.gradients.data[tokenIdx][dim] += gradient.data[pos][dim];
      }
    }
  }

  // Update weights using gradient descent
  update(learningRate) {
    for (let i = 0; i < this.vocabSize; i++) {
      for (let j = 0; j < this.embeddingDim; j++) {
        this.weights.data[i][j] -= learningRate * this.gradients.data[i][j];
      }
    }
  }

  // Visualize what a word means (its vector)
  visualizeWord(word, tokenizer) {
    const idx = tokenizer.getIndex(word);
    if (idx >= this.vocabSize) return null;
    
    console.log(`\n📊 Vector for "${word}":`);
    const vector = this.weights.data[idx];
    
    // Show first 10 dimensions
    console.log('First 10 dimensions:', vector.slice(0, 10).map(x => x.toFixed(3)).join(', '));
    
    // Calculate statistics
    const mean = vector.reduce((a, b) => a + b, 0) / vector.length;
    const variance = vector.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / vector.length;
    
    console.log(`Statistics: mean=${mean.toFixed(3)}, variance=${variance.toFixed(3)}`);
    
    return vector;
  }

  // Find similar words (cosine similarity)
  findSimilarWords(word, tokenizer, topK = 5) {
    const idx = tokenizer.getIndex(word);
    if (idx >= this.vocabSize) return [];
    
    const targetVec = this.weights.data[idx];
    
    // Compute cosine similarity with all words
    const similarities = [];
    for (let i = 0; i < this.vocabSize; i++) {
      if (i === idx) continue;
      
      const vec = this.weights.data[i];
      const dotProduct = targetVec.reduce((sum, val, j) => sum + val * vec[j], 0);
      const mag1 = Math.sqrt(targetVec.reduce((sum, val) => sum + val * val, 0));
      const mag2 = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
      
      const similarity = dotProduct / (mag1 * mag2);
      similarities.push({ index: i, similarity });
    }
    
    // Sort by similarity
    similarities.sort((a, b) => b.similarity - a.similarity);
    
    // Get top K
    return similarities.slice(0, topK).map(s => ({
      word: tokenizer.getWord(s.index),
      similarity: s.similarity.toFixed(3)
    }));
  }
}

