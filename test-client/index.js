const { default: axios } = require("axios");
let fs = require("fs");
let wav = require("node-wav");

let buffer = fs.readFileSync("test.wav");
let result = wav.decode(buffer);

console.log(result.sampleRate);
console.log(result.channelData);

axios.post("http://127.0.0.1:5000/upload", {
    }).then((resp) => {
        console.log(resp.data);
    }).catch((err) => {
        console.log(err);
    }
);