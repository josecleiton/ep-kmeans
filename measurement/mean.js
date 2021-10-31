var fs = require("fs");
const csv = require("csv-parser");

const filePath = "calibra/result.csv";

let time = [];

fs.createReadStream(filePath)
  .pipe(csv())
  .on("data", (line) => {
    time.push(line.time);
  })
  .on("end", () => {
    const mean = time
      .map((value) => value / time.length)
      .reduce((a, b) => a + b, 0)
      .toFixed(2);

    const deviation = Math.sqrt(
      time.map((time) => Math.pow(time - mean, 2)).reduce((a, b) => a + b) /
        time.length
    ).toFixed(2);

    console.log("Média: ", mean);
    console.log("Desvio: ", deviation);
    console.log("Razão: ", deviation / mean);
  });
