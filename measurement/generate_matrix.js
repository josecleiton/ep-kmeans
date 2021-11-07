var fs = require("fs");

const fileName = "matrix.txt";
const file = fs.createWriteStream(fileName);
const size = 500;

for (let i = 0; i < size; i++) {
  if (i !== 0) {
    file.write("\n");
  }
  for (let j = 0; j < size; j++) {
    file.write(`${Math.floor(Math.random() * 10)} `);
  }
}

file.close();
