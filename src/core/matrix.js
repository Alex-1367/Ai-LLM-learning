// src/core/matrix.js
export class Matrix {
  constructor(rows, cols, data = null) {
    this.rows = rows;
    this.cols = cols;
    
    if (data) {
      this.data = data;
    } else {
      // Xavier initialization (good for transformers)
      const scale = Math.sqrt(2.0 / (rows + cols));
      this.data = Array(rows).fill().map(() => 
        Array(cols).fill().map(() => (Math.random() * 2 - 1) * scale)
      );
    }
  }

  // Matrix multiplication - THIS IS WHERE THE MAGIC HAPPENS
  multiply(other) {
    if (this.cols !== other.rows) {
      throw new Error(`Cannot multiply ${this.rows}x${this.cols} with ${other.rows}x${other.cols}`);
    }
    
    const result = new Matrix(this.rows, other.cols);
    
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < other.cols; j++) {
        let sum = 0;
        for (let k = 0; k < this.cols; k++) {
          sum += this.data[i][k] * other.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    
    return result;
  }

  // Element-wise operations
  add(other) {
    if (this.rows !== other.rows || this.cols !== other.cols) {
      throw new Error('Matrix dimensions must match for addition');
    }
    
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j] + other.data[i][j];
      }
    }
    return result;
  }

  // Scalar operations
  multiplyScalar(scalar) {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = this.data[i][j] * scalar;
      }
    }
    return result;
  }

  // Apply function to each element
  map(func) {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = func(this.data[i][j], i, j);
      }
    }
    return result;
  }

  // Transpose
  transpose() {
    const result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[j][i] = this.data[i][j];
      }
    }
    return result;
  }

  // Softmax (for attention)
  softmax() {
    const result = new Matrix(this.rows, this.cols);
    
    for (let i = 0; i < this.rows; i++) {
      // Find max for numerical stability
      const maxVal = Math.max(...this.data[i]);
      
      // Compute exponentials
      let sum = 0;
      const expVals = this.data[i].map(x => {
        const exp = Math.exp(x - maxVal);
        sum += exp;
        return exp;
      });
      
      // Normalize
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = expVals[j] / sum;
      }
    }
    
    return result;
  }

  // Layer normalization
  layerNorm(epsilon = 1e-5) {
    const result = new Matrix(this.rows, this.cols);
    
    for (let i = 0; i < this.rows; i++) {
      // Compute mean and variance for this row
      const mean = this.data[i].reduce((a, b) => a + b, 0) / this.cols;
      const variance = this.data[i].reduce((a, b) => a + Math.pow(b - mean, 2), 0) / this.cols;
      
      // Normalize
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = (this.data[i][j] - mean) / Math.sqrt(variance + epsilon);
      }
    }
    
    return result;
  }

  // Print shape and first few values
  summary() {
    console.log(`Matrix shape: [${this.rows}, ${this.cols}]`);
    console.log('First 3x3:');
    for (let i = 0; i < Math.min(3, this.rows); i++) {
      console.log(this.data[i].slice(0, 3).map(x => x.toFixed(3)).join(' '));
    }
  }
}

// Activation functions
export const Activations = {
  relu: (x) => Math.max(0, x),
  reluDerivative: (x) => x > 0 ? 1 : 0,
  
  tanh: (x) => Math.tanh(x),
  tanhDerivative: (x) => 1 - Math.pow(Math.tanh(x), 2),
  
  gelu: (x) => { // Gaussian Error Linear Unit (used in modern LLMs)
    return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
  }
};

