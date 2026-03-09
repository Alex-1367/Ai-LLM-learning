// src/utils/pdfProcessor.js
import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import util from 'util';
import tesseract from 'node-tesseract-ocr';

const execPromise = util.promisify(exec);

export class PDFProcessor {
    constructor() {
        this.sentences = [];
        this.pdfStats = {
            totalFiles: 0,
            totalSentences: 0,
            filesProcessed: [],
            errors: []
        };

        // Tesseract config for German
        this.tesseractConfig = {
            lang: "deu",           // German language
            oem: 3,                 // Default OCR Engine Mode
            psm: 6,                 // Page segmentation mode: uniform block of text
            tessedit_char_whitelist: 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüßÄÖÜ0123456789.,!?;:-_ \'"()',
        };
    }

    // Recursively get all PDF files
    async getAllPDFFiles(dirPath) {
        let pdfFiles = [];

        try {
            const items = await fs.readdir(dirPath, { withFileTypes: true });

            for (const item of items) {
                const fullPath = path.join(dirPath, item.name);

                if (item.isDirectory()) {
                    // Recursively scan subdirectories
                    const subDirFiles = await this.getAllPDFFiles(fullPath);
                    pdfFiles = pdfFiles.concat(subDirFiles);
                } else if (item.isFile() && item.name.toLowerCase().endsWith('.pdf')) {
                    pdfFiles.push({
                        path: fullPath,
                        name: item.name,
                        folder: dirPath
                    });
                }
            }
        } catch (error) {
            console.error(`Error reading directory ${dirPath}:`, error.message);
        }

        return pdfFiles;
    }

    // Convert PDF to images using pdftoppm (if installed)
    async pdfToImages(pdfPath) {
        const tempDir = path.join('/tmp', 'pdf_ocr_' + Date.now());
        await fs.mkdir(tempDir, { recursive: true });

        try {
            // Use pdftoppm to convert PDF to images (high quality for OCR)
            await execPromise(`pdftoppm -png -r 300 "${pdfPath}" "${tempDir}/page"`);

            // Get all generated images
            const files = await fs.readdir(tempDir);
            const images = files
                .filter(f => f.endsWith('.png'))
                .map(f => path.join(tempDir, f))
                .sort(); // Ensure pages are in order

            return { images, tempDir };
        } catch (error) {
            // Fallback: if pdftoppm not installed, use a different method
            console.log('   ⚠️  pdftoppm not found, using alternative method...');
            return { images: [], tempDir };
        }
    }

    // Extract text using Tesseract OCR
    async extractTextWithTesseract(imagePath) {
        try {
            const text = await tesseract.recognize(imagePath, this.tesseractConfig);
            return text;
        } catch (error) {
            console.error(`   OCR Error on ${path.basename(imagePath)}:`, error.message);
            return '';
        }
    }

    // Main PDF processing function
    async processPDF(pdfInfo) {
        console.log(`   📄 Processing: ${pdfInfo.name}...`);

        let fullText = '';
        let images = [];
        let tempDir = null;

        try {
            // Method 1: Try pdftotext first (faster, if available)
            try {
                const { stdout } = await execPromise(`pdftotext "${pdfInfo.path}" -`);
                fullText = stdout;
            } catch {
                // If pdftotext fails or gives poor results, use OCR
                console.log(`   🔍 Using OCR for better German text extraction...`);

                // Convert PDF to images
                const result = await this.pdfToImages(pdfInfo.path);
                images = result.images;
                tempDir = result.tempDir;

                // OCR each image
                for (let i = 0; i < images.length; i++) {
                    console.log(`      Page ${i + 1}/${images.length}...`);
                    const pageText = await this.extractTextWithTesseract(images[i]);
                    fullText += pageText + '\n';
                }
            }

            // Clean the extracted text
            const cleanedText = this.cleanGermanText(fullText);

            // Split into sentences
            const sentences = this.extractGermanSentences(cleanedText);

            // Extract learning materials
            const dialogues = this.extractDialogues(cleanedText);
            const vocabulary = this.extractVocabulary(cleanedText);
            const grammarExamples = this.extractGrammarExamples(cleanedText);

            // Update stats
            this.pdfStats.totalFiles++;
            this.pdfStats.totalSentences += sentences.length;
            this.pdfStats.filesProcessed.push({
                name: pdfInfo.name,
                folder: pdfInfo.folder,
                sentences: sentences.length,
                dialogues: dialogues.length,
                vocabulary: vocabulary.length,
                pages: images.length || 1
            });

            console.log(`      ✅ Found ${sentences.length} sentences, ${dialogues.length} dialogues, ${vocabulary.length} new words`);

            return {
                sentences,
                dialogues,
                vocabulary,
                grammarExamples,
                fileName: pdfInfo.name,
                folder: pdfInfo.folder
            };

        } catch (error) {
            console.error(`      ❌ Error:`, error.message);
            this.pdfStats.errors.push({
                file: pdfInfo.name,
                error: error.message
            });
            return { sentences: [], dialogues: [], vocabulary: [], grammarExamples: {} };
        } finally {
            // Cleanup temp files
            if (tempDir) {
                try {
                    await fs.rm(tempDir, { recursive: true, force: true });
                } catch (e) {
                    // Ignore cleanup errors
                }
            }
        }
    }

    // Clean German text
    cleanGermanText(text) {
        return text
            .replace(/\f/g, ' ')              // Remove form feeds
            .replace(/\r\n/g, ' ')             // Normalize line endings
            .replace(/[|_]/g, '')              // Remove OCR artifacts
            .replace(/\s+/g, ' ')              // Normalize spaces
            .replace(/[„"']/g, '"')            // Normalize quotes
            .replace(/[–—]/g, '-')             // Normalize dashes
            .replace(/[ \t]+$/gm, '')          // Remove trailing spaces
            .replace(/^[ \t]+/gm, '')          // Remove leading spaces
            .replace(/([.!?])\s*(?=[A-Z])/g, '$1\n') // Add line breaks at sentence boundaries
            .trim();
    }

    // Extract German sentences
    extractGermanSentences(text) {
        // First, clean the text more aggressively
        const cleanText = text
            .replace(/\d+\.\s+/g, '')           // Remove numbered lists (1., 2., etc.)
            .replace(/Seite\s+\d+/gi, '')        // Remove page numbers
            .replace(/Übung\s+\d+/gi, '')        // Remove exercise numbers
            .replace(/Kapitel\s+\d+/gi, '')      // Remove chapter numbers
            .replace(/Lösung[\w\s]*/gi, '')      // Remove solution markers
            .replace(/\[.*?\]/g, '')              // Remove text in brackets
            .replace(/\(.*?\)/g, '')              // Remove text in parentheses (often instructions)
            .replace(/\s+/g, ' ')                 // Normalize spaces
            .trim();

        // Split by sentence endings (.!?) followed by space and capital letter
        const rawSentences = cleanText.split(/(?<=[.!?])\s+(?=[A-Zäöüß])/);

        return rawSentences
            .map(s => s.trim())
            .filter(s => {
                // Must have reasonable length
                if (s.length < 15 || s.length > 150) return false;

                // Must have at least 3 words
                const words = s.split(' ').filter(w => w.length > 0);
                if (words.length < 3) return false;

                // Must contain German letters
                if (!/[a-zA-ZäöüßÄÖÜ]/.test(s)) return false;

                // Must start with capital letter (German sentences start with capital)
                if (!/^[A-ZÄÖÜ]/.test(s)) return false;

                // Should not contain common OCR artifacts
                if (s.includes('|') || s.includes('_') || s.includes('*')) return false;

                // Should not be just numbers or exercise markers
                if (/^\d+[\s.]*$/.test(s)) return false;

                // Should have at least one German-specific character or common word
                const hasGermanChar = /[äöüß]/.test(s.toLowerCase());
                const hasGermanWord = /\b(der|die|das|und|ist|sind|ich|sie|wir|ihr)\b/.test(s.toLowerCase());

                return hasGermanChar || hasGermanWord;
            })
            .map(s => {
                // Clean up each sentence
                return s
                    .replace(/\s+/g, ' ')            // Normalize spaces
                    .replace(/\s+([,.!?])/g, '$1')   // Remove spaces before punctuation
                    .replace(/([,.!?])([A-Za-z])/g, '$1 $2') // Add space after punctuation if missing
                    .trim();
            });
    }

    // Add a new method to filter out exercise instructions
    filterExerciseInstructions(sentences) {
        const exercisePatterns = [
            /lösung/i,
            /übung/i,
            /seite\s+\d+/i,
            /antworten?/i,
            /beispiel/i,
            /ergänzen/i,
            /wählen/i,
            /schreiben/i,
            /lesen/i,
            /hören/i,
            /sprechen/i,
            /markieren/i,
            /notieren/i,
            /kapitel/i,
            /teil\s+\d+/i,
            /aktivität/i,
            /mögliche\s+antwort/i,
            /^[A-Z]\s*\d+\./,  // A1., B2., etc.
            /^\d+\.\s*[a-z]/,  // 1. a), 2. b), etc.
        ];

        return sentences.filter(sentence => {
            // Check if it's an exercise instruction
            const isExercise = exercisePatterns.some(pattern => pattern.test(sentence));
            if (isExercise) return false;

            // Check if it's too short or looks like a list item
            if (sentence.length < 20 && /^[A-Z]\s*\d/.test(sentence)) return false;

            return true;
        });
    }

    // Extract dialogues (conversational German)
    extractDialogues(text) {
        const patterns = [
            /[„"].*?[“"]/g,                    // Quoted speech
            /—[^—]+—/g,                         // Em-dash dialogues
            /[-–].*?[-–]/g,                      // Dash dialogues
            /[A-Z][a-zäöüß]+:\s+[^.!?]+[.!?]/g   // Speaker: dialogue
        ];

        let allDialogues = [];
        for (const pattern of patterns) {
            const matches = text.match(pattern) || [];
            allDialogues = allDialogues.concat(
                matches.map(d => d.replace(/[„"—–]/g, '').trim())
            );
        }

        return [...new Set(allDialogues)]
            .filter(d => d.length > 10 && d.length < 150)
            .slice(0, 100);
    }

    // Extract new vocabulary words
    extractVocabulary(text) {
        // Common German words to exclude
        const stopWords = new Set([
            'der', 'die', 'das', 'den', 'dem', 'des',
            'und', 'oder', 'aber', 'denn', 'sondern',
            'ist', 'sind', 'war', 'waren', 'wird', 'werden',
            'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr',
            'mein', 'dein', 'sein', 'ihr', 'unser', 'euer',
            'haben', 'hat', 'hatte', 'hatten',
            'können', 'kann', 'konnte', 'müssen', 'muss', 'musste',
            'gehen', 'geht', 'ging', 'kommen', 'kommt', 'kam',
            'sagen', 'sagt', 'sagte', 'machen', 'macht', 'machte',
            'sehen', 'sieht', 'sah', 'geben', 'gibt', 'gab',
            'ein', 'eine', 'einer', 'eines', 'einem', 'einen'
        ]);

        const words = text.split(/\s+/)
            .map(w => w.toLowerCase()
                .replace(/[.,!?;:()\[\]{}"'_\-]/g, '')
                .trim()
            )
            .filter(w => {
                // Basic filtering
                if (w.length < 3 || w.length > 20) return false;
                if (!/^[a-zäöüß]+$/.test(w)) return false;
                if (/^\d+$/.test(w)) return false;
                if (stopWords.has(w)) return false;

                // Should contain at least one vowel
                if (!/[aeiouäöü]/.test(w)) return false;

                return true;
            });

        // Count frequency
        const wordFreq = {};
        words.forEach(w => {
            wordFreq[w] = (wordFreq[w] || 0) + 1;
        });

        // Return words that appear at least twice (more likely to be real words)
        return Object.entries(wordFreq)
            .filter(([_, count]) => count >= 2)
            .map(([word]) => word)
            .slice(0, 500); // Limit to 500 new words per PDF
    }
    // Extract grammar patterns
    extractGrammarExamples(text) {
        return {
            questions: (text.match(/[A-Za-zäöüß][^.!?]*\?/g) || [])
                .filter(q => q.length < 100)
                .slice(0, 50),

            subordinateClauses: (text.match(/(weil|dass|wenn|ob|daß)[^.!?]*\./g) || [])
                .filter(c => c.length < 100)
                .slice(0, 50),

            separableVerbs: (text.match(/\b(auf|ab|an|aus|ein|mit|nach|vor|zu)\s+[^.!?]*\./g) || [])
                .filter(v => v.length < 100)
                .slice(0, 50),

            modalVerbs: (text.match(/\b(kann|muss|will|soll|darf|mag|können|müssen|wollen|sollen|dürfen|mögen)\s+[^.!?]*\./g) || [])
                .filter(m => m.length < 100)
                .slice(0, 50),

            nounPhrases: (text.match(/(der|die|das|dem|den|des)\s+[a-zäöüß]+\s+[a-zäöüß]+/g) || [])
                .slice(0, 50)
        };
    }

    // Process all PDFs in the Audio folder
    async processAudioFolder(basePath = '/home/admin/Videos/Audio') {
        console.log(`\n🔍 Scanning for German PDF books in: ${basePath}`);
        console.log('   (using Tesseract OCR with German language model)...\n');

        // Get all PDF files recursively
        const pdfFiles = await this.getAllPDFFiles(basePath);

        if (pdfFiles.length === 0) {
            console.log('   No PDF files found.');
            return [];
        }

        console.log(`   Found ${pdfFiles.length} PDF files total\n`);

        // Process files in batches to manage memory
        const batchSize = 3;
        let totalNewWords = 0;

        for (let i = 0; i < pdfFiles.length; i += batchSize) {
            const batch = pdfFiles.slice(i, i + batchSize);
            console.log(`\n📦 Processing batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(pdfFiles.length / batchSize)}`);

            const batchResults = await Promise.all(batch.map(async (pdfFile) => {
                const result = await this.processPDF(pdfFile);
                return result;
            }));

            // Combine results
            for (const result of batchResults) {
                if (result.sentences.length > 0) {
                    // Filter out exercise instructions
                    const goodSentences = this.filterExerciseInstructions(result.sentences);
                    this.sentences.push(...goodSentences);
                }
                if (result.vocabulary) {
                    totalNewWords += result.vocabulary.length;
                }
            }
        }

        console.log(`\n📊 Final Results:`);
        console.log(`   - Total new vocabulary words: ~${totalNewWords}`);

        // Print summary
        console.log('\n📊 PDF Processing Summary:');
        console.log(`   Total PDFs processed: ${this.pdfStats.totalFiles}`);
        console.log(`   Total sentences extracted: ${this.pdfStats.totalSentences}`);

        if (this.sentences.length > 0) {
            console.log('\n📝 Sample German sentences found:');
            const samples = this.sentences
                .sort(() => 0.5 - Math.random())
                .slice(0, 5);
            samples.forEach((s, i) => console.log(`   ${i + 1}. ${s}`));
        }

        return this.sentences;
    }

    // Get statistics
    getStats() {
        return {
            ...this.pdfStats,
            uniqueSentences: new Set(this.sentences).size,
            totalCharacters: this.sentences.reduce((sum, s) => sum + s.length, 0),
            foldersScanned: [...new Set(this.pdfStats.filesProcessed.map(f => f.folder))].length
        };
    }

    // Save report
    async saveReport(outputPath = './pdf_processing_report.json') {
        const report = {
            timestamp: new Date().toISOString(),
            stats: this.getStats(),
            filesProcessed: this.pdfStats.filesProcessed,
            errors: this.pdfStats.errors,
            sampleSentences: this.sentences.slice(0, 20)
        };

        await fs.writeFile(outputPath, JSON.stringify(report, null, 2));
        console.log(`\n📄 Report saved to: ${outputPath}`);

        return report;
    }
}