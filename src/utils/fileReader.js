// src/utils/fileReader.js
import { promises as fs } from 'fs';
import { basename, join } from 'path';

export default class GermanDataLoader {
  constructor() {
    this.vocabulary = new Map(); // German → English mapping
    this.texts = [];
    this.grammarPatterns = new Map();
  }

  // Parse your specific CSV format: "German word;English translation"
  async loadCSV(filePath) {
    const content = await fs.readFile(filePath, 'utf-8');
    const lines = content.split('\n').filter(line => line.trim());
    
    const pairs = [];
    for (let line of lines) {
      // Skip empty lines and comments
      if (line.startsWith('//') || !line.includes(';')) continue;
      
      const [german, english] = line.split(';').map(s => s.trim());
      if (german && english) {
        // Clean up the German word (remove grammatical markers)
        const cleanGerman = this.cleanGermanWord(german);
        pairs.push({
          german: cleanGerman,
          english: english,
          original: german // Keep original for grammar learning
        });
        
        // Store in vocabulary map
        this.vocabulary.set(cleanGerman, english);
      }
    }
    
    console.log(`Loaded ${pairs.length} word pairs from ${basename(filePath)}`);
    return pairs;
  }

  // Clean German words (remove grammatical annotations)
  cleanGermanWord(word) {
    // Remove things like "der/die", "/-nen", "Sg.", etc.
    return word
      .replace(/,.*$/, '')           // Remove comma and everything after
      .replace(/\s*\([^)]*\)/g, '')  // Remove parentheses content
      .replace(/\s*\/.*$/, '')        // Remove slashes and after
      .replace(/\s*-\s*\w+$/, '')     // Remove dash-suffixes
      .trim()
      .toLowerCase();
  }

  // Load OCR text files
  async loadTextFile(filePath) {
    const content = await fs.readFile(filePath, 'utf-8');
    
    // Clean OCR text (remove page numbers, fix common OCR errors)
    const cleanText = content
      .replace(/\f/g, ' ')           // Remove form feeds
      .replace(/\d+\s*$/gm, '')       // Remove page numbers
      .replace(/[|_]/g, '')           // Remove OCR artifacts
      .replace(/\s+/g, ' ')           // Normalize spaces
      .trim();
    
    // Split into sentences
    const sentences = cleanText.split(/[.!?]+/).filter(s => s.trim().length > 20);
    
    console.log(`Loaded ${sentences.length} sentences from ${basename(filePath)}`);
    return sentences;
  }

  // Load ALL your data
  async loadAllData() {
    console.log('📚 Loading German learning data...\n');
    
    // Define all your data paths
    const paths = {
      a1: {
        csv: [
          '/home/admin/Videos/Translate/A1/csv/',
          '/home/admin/Videos/Translate/A1/csv1/',
          '/home/admin/Videos/Translate/A1/csv2/',
          '/home/admin/Videos/Translate/A1/csv3/'
        ],
        txt: ['/home/admin/Videos/Translate/A1/txt/']
      },
      a2: {
        csv: [
          '/home/admin/Videos/Translate/A2/KB-CSV/',
          '/home/admin/Videos/Translate/A2/section/'
        ],
        txt: [
          '/home/admin/Videos/Translate/A2/AB-Image/',
          '/home/admin/Videos/Translate/A2/KB-Image/'
        ]
      }
    };

    // Load all CSV files (vocabulary)
    for (const level of ['a1', 'a2']) {
      for (const csvPath of paths[level].csv) {
        try {
          const files = await fs.readdir(csvPath);
          const csvFiles = files.filter(f => f.endsWith('.csv'));
          
          for (const file of csvFiles) {
            await this.loadCSV(join(csvPath, file));
          }
        } catch (err) {
          console.log(`Warning: Could not read ${csvPath} (${err.message})`);
        }
      }
    }

    // Load all TXT files (texts)
    for (const level of ['a1', 'a2']) {
      for (const txtPath of paths[level].txt) {
        try {
          const files = await fs.readdir(txtPath);
          const txtFiles = files.filter(f => f.endsWith('_ocr.txt'));
          
          for (const file of txtFiles) {
            const sentences = await this.loadTextFile(join(txtPath, file));
            this.texts.push(...sentences);
          }
        } catch (err) {
          console.log(`Warning: Could not read ${txtPath} (${err.message})`);
        }
      }
    }

    console.log(`\n✅ Total loaded:`);
    console.log(`   - ${this.vocabulary.size} German words with translations`);
    console.log(`   - ${this.texts.length} German sentences`);
    
    return {
      vocabulary: this.vocabulary,
      texts: this.texts
    };
  }

  // Extract grammar patterns from texts
  extractGrammarPatterns() {
    console.log('\n🔍 Extracting grammar patterns...');
    
    const patterns = {
      verbPosition: [],      // Track where verbs appear
      articleUsage: [],      // Track der/die/das usage
      sentenceStructure: []  // Track sentence patterns
    };

    for (const text of this.texts) {
      const words = text.split(' ');
      
      // Find verbs (simplified - look for common verb endings)
      const verbs = words.filter(w => 
        w.endsWith('en') || w.endsWith('t') || w.endsWith('e')
      );
      
      if (verbs.length > 0) {
        patterns.verbPosition.push({
          sentence: text,
          verbIndex: words.findIndex(w => verbs.includes(w))
        });
      }
      
      // Find articles
      const articles = words.filter(w => 
        ['der', 'die', 'das', 'dem', 'den', 'des', 'ein', 'eine'].includes(w.toLowerCase())
      );
      
      if (articles.length > 0) {
        patterns.articleUsage.push({
          sentence: text,
          articles: articles
        });
      }
    }

    console.log(`   Found ${patterns.verbPosition.length} verb patterns`);
    console.log(`   Found ${patterns.articleUsage.length} article patterns`);
    
    return patterns;
  }
}

