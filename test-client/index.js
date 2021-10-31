let axios = require("axios");
let fs = require("fs");
var FormData = require('form-data');
let wav = require("node-wav");

let buffer = fs.readFileSync("test.wav");

console.log(buffer);

let formData = new FormData();
formData.append('wavfile', buffer, "recording.txt");

axios.post("http://127.0.0.1:5000/upload",
    formData, {
    headers: {
        'Content-Type': `multipart/form-data; boundary=${data._boundary}`,
    },
}
)