// index.js
import GermanDataLoader from './src/utils/fileReader.js';
import VocabularyBuilder from './data/vocabulary.js';
import TinyLLM from './src/core/model.js';
import FillInBlankExercise from './src/features/fillInBlank.js';
import { ModelSaver } from './src/utils/modelSaver.js';

async function main() {
    console.log('🇩🇪 German Learning LLM Project\n');
    console.log('Goal: Understand how LLMs work while learning German\n');

    // Step 1: Load your data
    console.log('📂 Step 1: Loading German learning data...');
    const dataLoader = new GermanDataLoader();
    const germanData = await dataLoader.loadAllData();

    // Step 2: Build vocabulary
    console.log('\n📖 Step 2: Building vocabulary...');
    const vocabulary = new VocabularyBuilder(germanData);

    // Step 3: Extract grammar patterns
    console.log('\n🔍 Step 3: Analyzing grammar patterns...');
    const patterns = dataLoader.extractGrammarPatterns();

    // Step 4: Create the LLM
    console.log('\n🏗️  Step 4: Building TinyLLM from scratch...');
    const config = {
        embeddingDim: 32,  // Small for learning
        numLayers: 2,       // 2 transformer blocks
        numHeads: 4,        // 4 attention heads
        maxLength: 32
    };

    const model = new TinyLLM(config, vocabulary);

    // Step 5: Show how words become vectors
    console.log('\n🔤 Step 5: Word Embeddings Demonstration');
    console.log('Showing how words are represented as vectors:');

    const sampleWords = ['haus', 'mann', 'frau', 'kind', 'gehen'];
    model.visualizeWordVectors(sampleWords);

    // Step 6: Demonstrate attention
    console.log('\n🎯 Step 6: Attention Mechanism Demonstration');
    console.log('Showing how the model pays attention to different words:');

    const sampleSentence = "der mann geht nach hause";
    const tokens = vocabulary.encode(sampleSentence);
    const embeddings = model.embedding.forward(tokens);

    // Run through first transformer layer to see attention
    const firstLayer = model.layers[0];
    firstLayer.attention.forward(embeddings);
    firstLayer.attention.visualizeAttention(sampleSentence.split(' '));

    // Step 7: Try text generation
    console.log('\n💭 Step 7: Text Generation');
    console.log('The model tries to complete sentences:');

    const prompts = ['ich', 'der mann', 'sie hat', 'wir gehen'];
    for (const prompt of prompts) {
        model.generate(prompt, 10, 0.8);
    }

    // Step 8: Learning exercises
    console.log('\n📝 Step 8: German Learning Exercises');
    console.log('Using the model to help learn German:');

    const exercise = new FillInBlankExercise(model, vocabulary);
    await exercise.runExercise(germanData.texts);

    // Step 9: Save project
    console.log('\n💾 Step 9: Saving your trained model...');
    const modelSaver = new ModelSaver(model, vocabulary);
    const savedModelPath = await modelSaver.saveAsGGUFAnalog();

    console.log('\n🎓 What we learned:');
    // ... rest of your output

    console.log(`\n✅ Project complete! Your TinyLLM model is saved at:`);
    console.log(`   ${savedModelPath}`);
    console.log(`\n🔍 To explore your model files:`);
    console.log(`   cd ${savedModelPath}`);
    console.log(`   ls -la`);

    // Step 9: What we learned
    console.log('\n🎓 What we learned:');
    console.log('1. Words become vectors (embeddings)');
    console.log('2. Attention lets words "look at" each other');
    console.log('3. Transformers process sequences layer by layer');
    console.log('4. The model learns patterns from your German texts');
    console.log('5. Generation works by predicting one word at a time');

    console.log('\n✅ Project complete! Your TinyLLM understands:');
    console.log(`   - ${vocabulary.vocabSize} German words`);
    console.log(`   - Trained on ${germanData.texts.length} sentences`);
    console.log(`   - Using ${config.embeddingDim}-dimensional vectors`);
}

main().catch(console.error);