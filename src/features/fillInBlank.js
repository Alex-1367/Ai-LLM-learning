// src/features/fillInBlank.js
export default class FillInBlankExercise {
  constructor(model, tokenizer) {
    this.model = model;
    this.tokenizer = tokenizer;
  }

  // Generate fill-in-blank exercise
  createExercise(sentence, maskWord = null) {
    const words = sentence.split(' ');
    
    if (maskWord) {
      // Mask specific word
      const index = words.findIndex(w => w.toLowerCase() === maskWord.toLowerCase());
      if (index >= 0) {
        const answer = words[index];
        words[index] = '_____';
        return {
          exercise: words.join(' '),
          answer: answer,
          hint: this.tokenizer.getTranslation(answer)
        };
      }
    }
    
    // Randomly mask a content word
    const contentWords = words.filter(w => 
      w.length > 3 && !['der', 'die', 'das', 'und', 'ist'].includes(w.toLowerCase())
    );
    
    if (contentWords.length > 0) {
      const targetWord = contentWords[Math.floor(Math.random() * contentWords.length)];
      const index = words.findIndex(w => w === targetWord);
      const answer = words[index];
      words[index] = '_____';
      
      return {
        exercise: words.join(' '),
        answer: answer,
        hint: this.tokenizer.getTranslation(answer)
      };
    }
    
    return null;
  }

  // Let model predict the missing word
  async predictMissingWord(sentence, missingWordIndex) {
    const words = sentence.split(' ');
    const context = words.map((w, i) => i === missingWordIndex ? '<MASK>' : w).join(' ');
    
    // Use model to predict
    const tokens = this.tokenizer.encode(context);
    const logits = this.model.forward(tokens);
    
    // Get prediction for masked position
    const maskPos = tokens.findIndex(t => t === this.tokenizer.specialTokens['<MASK>']);
    if (maskPos >= 0) {
      const predictions = logits.data[maskPos];
      const topK = this.getTopKPredictions(predictions, 5);
      
      return topK.map(p => ({
        word: this.tokenizer.getWord(p.index),
        confidence: (p.prob * 100).toFixed(1) + '%',
        translation: this.tokenizer.getTranslation(this.tokenizer.getWord(p.index))
      }));
    }
    
    return [];
  }

  getTopKPredictions(probs, k) {
    return probs
      .map((prob, index) => ({ index, prob }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, k);
  }

  // Interactive exercise
  async runExercise(sentences) {
    console.log('\n📝 Fill in the Blank Exercise\n');
    
    for (let i = 0; i < Math.min(5, sentences.length); i++) {
      const sentence = sentences[Math.floor(Math.random() * sentences.length)];
      const exercise = this.createExercise(sentence);
      
      if (exercise) {
        console.log(`\n${i + 1}. ${exercise.exercise}`);
        console.log(`Hint: ${exercise.hint}`);
        
        // Simulate user input (in real app, you'd get from stdin)
        const userAnswer = '...'; // Placeholder
        
        // Show model's predictions
        const predictions = await this.predictMissingWord(
          exercise.exercise.replace(/_____/g, '<MASK>'), 
          0
        );
        
        console.log('Model predicts:');
        predictions.forEach((p, i) => {
          console.log(`   ${i + 1}. ${p.word} (${p.confidence}) - ${p.translation}`);
        });
        
        console.log(`Correct answer: ${exercise.answer}`);
      }
    }
  }
}
