let axios = require("axios");
let fs = require("fs");
var FormData = require('form-data');
let wav = require("node-wav");

let buffer = fs.readFileSync("test.wav");

console.log(buffer);

let formData = new FormData();
formData.append('wavfile', buffer);

axios.post("http://127.0.0.1:5000/upload", 
    { 
        data: formData,
        timestamp: Date.now(),
    }).then((resp) => {
        console.log(resp.data);
    }).catch((err) => {
        console.log(err);
    }
);