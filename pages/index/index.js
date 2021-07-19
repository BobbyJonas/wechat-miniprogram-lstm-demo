// index.js
const regeneratorRuntime = require('regenerator-runtime');
const tf = require('@tensorflow/tfjs-core');
const tfl = require('@tensorflow/tfjs-layers');
const _dictionary = require('./dictionary')

Page({
  data: {
    inputWord: '蕭峯吃了一驚，心想：「哥哥大喜之餘，說話有些忘形了，眼下亂成',
    outputWord: '',
  },
  async onReady() {
    // 加载模型
    this.net = await tfl.loadLayersModel('http://localhost/lstm2/model.json');
  },
  clickHandler() {
    wx.showLoading({
      title: '玩命生成中',
      mask: true,		
    });
    const generateLength = 150;  // 生成几个字
    const temperature = 0.6;   // 随机度
    const output = this.genarateText(
      this.net,
      this.data.inputWord,
      generateLength, 
      temperature,
    );
    this.setData({
      outputWord: this.data.inputWord + output,
    });
    wx.hideLoading();
  },
  sample(probs) {
    return tf.tidy(() => {
      const temperature = parseFloat(0.6);
      const logits = tf.div(probs, Math.max(temperature, 1e-6));
      const isNormalized = false;
      const seed = null;
      const generatedIndices = tf.multinomial(logits, 1, seed, isNormalized).dataSync();
      return generatedIndices[generatedIndices.length - 1];
    });
  },
  genarateText(model, input, generateLength, temperature) {
    let seedSentenceIndices = [];
    let seedSentence = input;
  
    if (seedSentence.length == 0) {
      seedSentence = initializeSeed();
    } else if (seedSentence.length > 30) {
      seedSentence = seedSentence.slice(seedSentence.length - 30, seedSentence.length);
    }
  
    for (let i = 0; i < seedSentence.length; ++i) {
      seedSentenceIndices.push(_dictionary.WORD2INDEX[seedSentence[i]]);
    }
  
    let generated = '';
    let inputEval = tf.tensor(seedSentenceIndices);
    model.resetStates();
  
    while (generated.length < generateLength) {
      var input = inputEval;
      input = tf.expandDims(inputEval, 0);
      const output = model.predict(input);
      const sampledIndex = this.sample(tf.squeeze(output), temperature);
      const sampledChar = _dictionary.INDEX2WORD[sampledIndex];
  
      generated += sampledChar;
  
      seedSentenceIndices = seedSentenceIndices.slice(1);
      seedSentenceIndices.push(sampledIndex);
      inputEval = tf.tensor(seedSentenceIndices);
      input.dispose();
      output.dispose();
    }
    return generated;
  }
})