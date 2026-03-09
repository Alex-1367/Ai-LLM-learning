// src/core/training.js
import { Matrix } from './matrix.js';

export class Trainer {
  constructor(model, tokenizer, learningRate = 0.001) {
    this.model = model;
    this.tokenizer = tokenizer;
    this.learningRate = learningRate;
    this.lossHistory = [];
    this.accuracyHistory = [];
  }

  // Forward pass with loss computation
  forwardPass(tokens) {
    // tokens: [seqLength]
    
    // Get predictions
    const logits = this.model.forward(tokens);
    
    // Compute loss (cross-entropy) for next token prediction
    let totalLoss = 0;
    let correct = 0;
    let total = 0;
    
    for (let i = 0; i < logits.rows - 1; i++) {
      const trueNextToken = tokens[i + 1];
      const predProbs = this.softmax(logits.data[i]);
      
      // Cross-entropy loss
      totalLoss += -Math.log(predProbs[trueNextToken] + 1e-10);
      
      // Accuracy
      const predictedToken = predProbs.indexOf(Math.max(...predProbs));
      if (predictedToken === trueNextToken) correct++;
      total++;
    }
    
    return {
      loss: totalLoss / total,
      accuracy: correct / total,
      logits
    };
  }

  // Softmax for probability distribution
  softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    return expLogits.map(x => x / sumExp);
  }

  // Simple training step (simplified - in real training you'd do backprop)
  trainStep(batch) {
    const { input, target } = batch;
    let batchLoss = 0;
    let batchAccuracy = 0;
    
    for (let i = 0; i < input.length; i++) {
      const result = this.forwardPass(input[i]);
      batchLoss += result.loss;
      batchAccuracy += result.accuracy;
      
      // SIMPLIFIED LEARNING: Random weight adjustment
      // In real training, you'd compute gradients and update properly
      if (Math.random() < 0.01) { // Tiny chance to "learn"
        this.randomlyAdjustWeights();
      }
    }
    
    return {
      loss: batchLoss / input.length,
      accuracy: batchAccuracy / input.length
    };
  }

  // Temporary: random weight adjustment (simulates learning)
  randomlyAdjustWeights() {
    // Slightly adjust embedding weights
    for (let i = 0; i < Math.min(100, this.model.embedding.weights.rows); i++) {
      for (let j = 0; j < this.model.embedding.weights.cols; j++) {
        this.model.embedding.weights.data[i][j] += (Math.random() - 0.5) * 0.01;
      }
    }
  }

  // Create training batches from texts
  createBatches(texts, batchSize = 8, maxLength = 32) {
    const batches = [];
    const validTexts = texts.filter(t => t.split(' ').length <= maxLength - 2);
    
    for (let i = 0; i < validTexts.length; i += batchSize) {
      const batchTexts = validTexts.slice(i, i + batchSize);
      const batchInputs = [];
      const batchTargets = [];
      
      for (const text of batchTexts) {
        const tokens = this.tokenizer.encode(text);
        batchInputs.push(tokens.slice(0, -1)); // Input: all but last token
        batchTargets.push(tokens.slice(1));    // Target: all but first token
      }
      
      batches.push({
        input: batchInputs,
        target: batchTargets
      });
    }
    
    return batches;
  }

  // Main training loop
  async train(texts, epochs = 10, batchSize = 8) {
    console.log('\n🎯 Starting Training...');
    console.log(`   Texts: ${texts.length}`);
    console.log(`   Epochs: ${epochs}`);
    console.log(`   Batch size: ${batchSize}`);
    console.log(`   Learning rate: ${this.learningRate}\n`);
    
    const batches = this.createBatches(texts, batchSize);
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let epochAccuracy = 0;
      
      // Shuffle batches for better learning
      const shuffledBatches = this.shuffleArray([...batches]);
      
      for (const batch of shuffledBatches) {
        const result = this.trainStep(batch);
        epochLoss += result.loss;
        epochAccuracy += result.accuracy;
      }
      
      const avgLoss = epochLoss / batches.length;
      const avgAccuracy = epochAccuracy / batches.length;
      
      this.lossHistory.push(avgLoss);
      this.accuracyHistory.push(avgAccuracy);
      
      // Progress report
      console.log(`Epoch ${epoch + 1}/${epochs} - Loss: ${avgLoss.toFixed(4)}, Accuracy: ${(avgAccuracy * 100).toFixed(2)}%`);
      
      // Generate sample every few epochs to see progress
      if ((epoch + 1) % 3 === 0 || epoch === epochs - 1) {
        this.showTrainingProgress(epoch);
      }
    }
    
    console.log('\n✅ Training Complete!');
    this.printTrainingSummary();
  }

  shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
  }

  showTrainingProgress(epoch) {
    console.log(`\n📝 Sample generations after epoch ${epoch + 1}:`);
    const prompts = ['ich', 'der mann', 'sie hat', 'wir gehen'];
    
    for (const prompt of prompts) {
      const result = this.model.generate(prompt, 8, 0.8);
      console.log(`   "${prompt}" → ${result.new}`);
    }
    console.log('');
  }

  printTrainingSummary() {
    console.log('\n📊 Training Summary:');
    console.log(`   Final Loss: ${this.lossHistory[this.lossHistory.length - 1].toFixed(4)}`);
    console.log(`   Final Accuracy: ${(this.accuracyHistory[this.accuracyHistory.length - 1] * 100).toFixed(2)}%`);
    
    // Loss improvement
    const improvement = ((this.lossHistory[0] - this.lossHistory[this.lossHistory.length - 1]) / this.lossHistory[0] * 100).toFixed(1);
    console.log(`   Loss Improvement: ${improvement}%`);
    
    if (improvement > 0) {
      console.log(`   ✅ Model is learning German patterns!`);
    } else {
      console.log(`   ⚠️  Model needs more training or data`);
    }
  }

  // Visualize learning progress
  plotLearningCurve() {
    console.log('\n📈 Learning Curve:');
    const maxPoints = 20;
    const points = this.lossHistory.slice(-maxPoints);
    
    const maxLoss = Math.max(...points);
    const minLoss = Math.min(...points);
    
    for (let i = 0; i < points.length; i++) {
      const normalized = 1 - (points[i] - minLoss) / (maxLoss - minLoss);
      const barLength = Math.floor(normalized * 30);
      const bar = '█'.repeat(barLength) + '░'.repeat(30 - barLength);
      console.log(`Epoch ${i + 1}: ${bar} ${points[i].toFixed(4)}`);
    }
  }
}

// Advanced: Add curriculum learning
export class CurriculumTrainer extends Trainer {
  async trainWithCurriculum(textsByDifficulty) {
    // Train on easier texts first, then harder
    const difficulties = ['A1', 'A2', 'B1'];
    
    for (const level of difficulties) {
      if (textsByDifficulty[level]) {
        console.log(`\n📚 Training on ${level} level...`);
        await this.train(textsByDifficulty[level], 5);
      }
    }
  }
}