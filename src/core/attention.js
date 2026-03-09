// src/core/attention.js
import { Matrix } from './matrix.js';

export default class MultiHeadAttention {
  constructor(embeddingDim, numHeads = 4) {
    this.embeddingDim = embeddingDim;
    this.numHeads = numHeads;
    this.headDim = embeddingDim / numHeads;
    
    console.log(`🎯 Creating Multi-Head Attention with ${numHeads} heads`);
    
    // Query, Key, Value projection matrices
    // These transform embeddings into different representations
    this.Wq = new Matrix(embeddingDim, embeddingDim);
    this.Wk = new Matrix(embeddingDim, embeddingDim);
    this.Wv = new Matrix(embeddingDim, embeddingDim);
    this.Wo = new Matrix(embeddingDim, embeddingDim); // Output projection
    
    // Gradients
    this.gradWq = new Matrix(embeddingDim, embeddingDim);
    this.gradWk = new Matrix(embeddingDim, embeddingDim);
    this.gradWv = new Matrix(embeddingDim, embeddingDim);
    this.gradWo = new Matrix(embeddingDim, embeddingDim);
    
    // For visualization
    this.lastAttentionWeights = null;
  }

  // Forward pass - THIS IS WHERE ATTENTION HAPPENS
  forward(x) {
    // x shape: [seqLength, embeddingDim]
    this.lastInput = x;
    const seqLength = x.rows;
    
    // 1. Project to queries, keys, values
    // Queries: What am I looking for?
    // Keys: What do I contain?
    // Values: What information do I pass on?
    const Q = x.multiply(this.Wq);
    const K = x.multiply(this.Wk);
    const V = x.multiply(this.Wv);
    
    // 2. Split into multiple heads (each head learns different patterns)
    const Qheads = this.splitHeads(Q);
    const Kheads = this.splitHeads(K);
    const Vheads = this.splitHeads(V);
    
    // 3. Compute attention for each head
    const attentionOutputs = [];
    const attentionWeights = [];
    
    for (let h = 0; h < this.numHeads; h++) {
      // Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
      
      // QK^T: How much does each word attend to every other word?
      const scores = Qheads[h].multiply(Kheads[h].transpose());
      
      // Scale by sqrt(headDim) to prevent extreme values
      const scaled = scores.map(x => x / Math.sqrt(this.headDim));
      
      // Softmax to get attention weights (probabilities)
      const weights = scaled.softmax();
      
      // Apply attention to values
      const headOutput = weights.multiply(Vheads[h]);
      
      attentionOutputs.push(headOutput);
      attentionWeights.push(weights);
    }
    
    // Save for visualization
    this.lastAttentionWeights = attentionWeights[0]; // First head's attention
    
    // 4. Concatenate all heads
    const concatenated = this.concatHeads(attentionOutputs);
    
    // 5. Final output projection
    const output = concatenated.multiply(this.Wo);
    
    return output;
  }

  splitHeads(matrix) {
    // Split embedding dimension into numHeads
    const heads = [];
    for (let h = 0; h < this.numHeads; h++) {
      const start = h * this.headDim;
      const end = (h + 1) * this.headDim;
      
      const headData = matrix.data.map(row => row.slice(start, end));
      heads.push(new Matrix(matrix.rows, this.headDim, headData));
    }
    return heads;
  }

  concatHeads(heads) {
    const seqLength = heads[0].rows;
    const concatData = Array(seqLength).fill().map(() => []);
    
    for (let h = 0; h < this.numHeads; h++) {
      for (let i = 0; i < seqLength; i++) {
        concatData[i].push(...heads[h].data[i]);
      }
    }
    
    return new Matrix(seqLength, this.embeddingDim, concatData);
  }

  // Visualize attention patterns
  visualizeAttention(words) {
    if (!this.lastAttentionWeights) {
      console.log('No attention weights available. Run forward pass first.');
      return;
    }
    
    console.log('\n🔍 Attention Pattern Visualization:');
    console.log('Words:', words.join(' '));
    console.log('\nAttention Matrix (how much each word attends to others):');
    
    const weights = this.lastAttentionWeights;
    for (let i = 0; i < Math.min(5, weights.rows); i++) {
      const row = weights.data[i].slice(0, Math.min(5, weights.cols));
      console.log(`${words[i]}: ${row.map(x => x.toFixed(2)).join(' ')}`);
    }
    
    // Find the strongest attention
    let maxI = 0, maxJ = 0, maxVal = -1;
    for (let i = 0; i < weights.rows; i++) {
      for (let j = 0; j < weights.cols; j++) {
        if (weights.data[i][j] > maxVal) {
          maxVal = weights.data[i][j];
          maxI = i;
          maxJ = j;
        }
      }
    }
    
    console.log(`\nStrongest attention: "${words[maxI]}" → "${words[maxJ]}" (${maxVal.toFixed(2)})`);
  }
}

