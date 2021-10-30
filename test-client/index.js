let fs = require("fs");
let wav = require("node-wav");

let buffer = fs.readFileSync("test.wav");
let result = wav.decode(buffer);

console.log(result.sampleData);
console.log(result.channelData);