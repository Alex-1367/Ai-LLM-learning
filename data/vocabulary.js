// src/data/vocabulary.js
export default class VocabularyBuilder {
  constructor(germanData) {
    this.wordToIndex = new Map();
    this.indexToWord = new Map();
    this.wordToTranslation = new Map(); // German → English
    this.translationToWord = new Map(); // English → German
    
    // Special tokens
    this.specialTokens = {
      '<PAD>': 0,
      '<UNK>': 1,
      '<BOS>': 2,
      '<EOS>': 3,
      '<MASK>': 4  // For fill-in-blank exercises
    };
    
    this.buildFromData(germanData);
  }

  buildFromData(germanData) {
    let index = Object.keys(this.specialTokens).length;
    
    // Add all German words from vocabulary
    for (let [german, english] of germanData.vocabulary) {
      if (!this.wordToIndex.has(german)) {
        this.wordToIndex.set(german, index);
        this.indexToWord.set(index, german);
        this.wordToTranslation.set(german, english);
        index++;
      }
    }
    
    // Also add words from texts that might not be in vocabulary
    for (const text of germanData.texts) {
      const words = text.toLowerCase().split(/\s+/);
      for (const word of words) {
        if (!this.wordToIndex.has(word) && word.length > 1) {
          this.wordToIndex.set(word, index);
          this.indexToWord.set(index, word);
          index++;
        }
      }
    }
    
    this.vocabSize = index;
    console.log(`📖 Built vocabulary with ${this.vocabSize} words`);
  }

  // Get word index (token)
  getIndex(word) {
    return this.wordToIndex.get(word.toLowerCase()) || this.specialTokens['<UNK>'];
  }

  // Get word from index
  getWord(index) {
    return this.indexToWord.get(index) || '<UNK>';
  }

  // Get English translation
  getTranslation(german) {
    return this.wordToTranslation.get(german) || '?';
  }

  // Get German word from English
  getGerman(english) {
    for (let [german, trans] of this.wordToTranslation) {
      if (trans.includes(english)) return german;
    }
    return null;
  }

  // Encode sentence to tokens
  encode(sentence, addSpecialTokens = true) {
    const words = sentence.toLowerCase().split(/\s+/);
    const tokens = [];
    
    if (addSpecialTokens) tokens.push(this.specialTokens['<BOS>']);
    
    for (const word of words) {
      tokens.push(this.getIndex(word));
    }
    
    if (addSpecialTokens) tokens.push(this.specialTokens['<EOS>']);
    
    return tokens;
  }

  // Decode tokens to sentence
  decode(tokens, skipSpecial = true) {
    return tokens
      .filter(t => !skipSpecial || t >= Object.keys(this.specialTokens).length)
      .map(t => this.getWord(t))
      .join(' ');
  }

  // Create masked version for exercises
  createMaskedSentence(sentence, maskProb = 0.15) {
    const words = sentence.split(' ');
    const masked = [];
    const answers = [];
    
    for (const word of words) {
      if (Math.random() < maskProb && this.wordToIndex.has(word.toLowerCase())) {
        masked.push('<MASK>');
        answers.push(word);
      } else {
        masked.push(word);
      }
    }
    
    return {
      masked: masked.join(' '),
      answers: answers
    };
  }
}

