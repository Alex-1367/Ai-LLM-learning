// index.js
import GermanDataLoader from './src/utils/fileReader.js';
import { PDFProcessor } from './src/utils/pdfProcessor.js';
import VocabularyBuilder from './data/vocabulary.js';
import TinyLLM from './src/core/model.js';
import { Trainer } from './src/core/training.js';
import { ModelSaver } from './src/utils/modelSaver.js';
import FillInBlankExercise from './src/features/fillInBlank.js';
import path from 'path';

async function main() {
    console.log('🇩🇪 German Learning LLM Project with Training\n');

    // Step 1: Load your German data
    console.log('📂 Step 1: Loading German learning data...');
    const dataLoader = new GermanDataLoader();
    let germanData = await dataLoader.loadAllData();


    // Step 2: Load PDF books from Audio folder recursively
    console.log('\n📚 Step 2: Loading German PDF books from Audio folder...');
    const pdfProcessor = new PDFProcessor();

    // Process ALL PDFs in /home/admin/Videos/Audio and subfolders
    const pdfSentences = await pdfProcessor.processAudioFolder('/home/admin/Videos/Audio');

    if (pdfSentences.length > 0) {
        console.log(`\n✅ Added ${pdfSentences.length} sentences from PDFs`);

        // Save a report of what was found
        await pdfProcessor.saveReport('./pdf_processing_report.json');

        // Add to your training data
        const originalCount = germanData.texts.length;
        germanData.texts.push(...pdfSentences);

        // Show statistics with proper error handling
        const stats = pdfProcessor.getStats();
        console.log('\n📊 PDF Statistics:');
        console.log(`   - Files processed: ${stats.totalFiles || 0}`);
        console.log(`   - Sentences extracted: ${stats.totalSentences || 0}`);
        console.log(`   - Unique sentences: ${stats.uniqueSentences || 0}`);
        if (stats.averageSentenceLength) {
            console.log(`   - Average sentence length: ${stats.averageSentenceLength.toFixed(1)} words`);
        }
        console.log(`   - Folders scanned: ${stats.foldersScanned || 0}`);

        console.log(`\n📚 Total training data now: ${germanData.texts.length} sentences`);
    } else {
        console.log('\n⚠️  No German sentences could be extracted from PDFs.');
        console.log('   Check the pdf_processing_report.json for details.');
    }

    // Step 3: Build vocabulary
    console.log('\n📖 Step 3: Building vocabulary...');
    const vocabulary = new VocabularyBuilder(germanData);

    // Step 4: Create the LLM
    console.log('\n🏗️ Step 4: Building TinyLLM...');
    const config = {
        embeddingDim: 64,  // Increased for better learning
        numLayers: 3,       // More layers for complex patterns
        numHeads: 4,
        maxLength: 48
    };

    const model = new TinyLLM(config, vocabulary);
    model.totalSentences = germanData.texts.length;

    // Step 5: Train the model!
    console.log('\n🎯 Step 5: Training the model on your German texts...');
    const trainer = new Trainer(model, vocabulary, 0.01);

    // Train on your texts
    await trainer.train(germanData.texts, 15, 8);

    // Show learning curve
    trainer.plotLearningCurve();

    // Step 6: Test what it learned
    console.log('\n🔤 Step 6: Testing what the model learned...');

    // Check word vectors after training
    console.log('\nWord vectors after training:');
    const testWords = ['haus', 'mann', 'frau', 'kind', 'gehen', 'essen', 'trinken'];
    model.visualizeWordVectors(testWords);

    // Check attention patterns
    console.log('\n🎯 Attention patterns after training:');
    const testSentence = "der mann geht nach hause";
    const tokens = vocabulary.encode(testSentence);
    const embeddings = model.embedding.forward(tokens);
    const firstLayer = model.layers[0];
    firstLayer.attention.forward(embeddings);
    firstLayer.attention.visualizeAttention(testSentence.split(' '));

    // Step 7: Generate better text!
    console.log('\n💭 Step 7: Text generation after training:');
    const prompts = [
        'ich',
        'der mann',
        'sie hat',
        'wir gehen',
        'heute ist',
        'mein name ist'
    ];

    for (const prompt of prompts) {
        model.generate(prompt, 12, 0.7);
    }

    // Step 8: Learning exercises with trained model
    console.log('\n📝 Step 8: German Exercises with trained model:');
    const exercise = new FillInBlankExercise(model, vocabulary);

    // Use some of your actual sentences for exercises
    const sampleSentences = germanData.texts.slice(0, 10);
    await exercise.runExercise(sampleSentences);

    // Step 9: Save the trained model
    console.log('\n💾 Step 9: Saving your trained model...');
    const modelSaver = new ModelSaver(model, vocabulary);
    const modelPath = await modelSaver.saveAsGGUFAnalog();

    // Final summary
    console.log('\n🎓 What we accomplished:');
    console.log('1. ✅ Loaded your German vocabulary and texts');
    console.log('2. ✅ Built a transformer LLM from scratch');
    console.log('3. ✅ Trained it on real German sentences');
    console.log('4. ✅ Model learned word relationships (embeddings)');
    console.log('5. ✅ Model learned attention patterns');
    console.log('6. ✅ Can generate German text');
    console.log('7. ✅ Can help with fill-in-blank exercises');

    console.log(`\n📊 Final Model Statistics:`);
    console.log(`   Vocabulary: ${vocabulary.vocabSize} words`);
    console.log(`   Training texts: ${germanData.texts.length} sentences`);
    console.log(`   Embedding dimensions: ${config.embeddingDim}`);
    console.log(`   Transformer layers: ${config.numLayers}`);
    console.log(`   Total parameters: ~${modelSaver.calculateParameters().toLocaleString()}`);

    console.log(`\n💾 Model saved at: ${modelPath}`);
    console.log(`\n✨ Your German LLM is ready! Run it again to continue training.`);
}

main().catch(console.error);