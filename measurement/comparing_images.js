const fsPromises = require("fs").promises;
var fs = require("fs");
const Jimp = require("jimp");

async function getFiles(dir, files) {
  if (!files) files = [];

  let listFiles = await fsPromises.readdir(dir);
  for (let k in listFiles) {
    let stat = await fsPromises.stat(dir + "/" + listFiles[k]);
    if (stat.isDirectory()) await getFiles(dir + "/" + listFiles[k], files);
    else files.push(dir + "/" + listFiles[k]);
  }

  return files;
}

function getRandomArbitrary(min, max) {
  const value = Math.random() * (max - min) + min;
  return Math.round(value);
}

async function compareImage() {
  const resolutions = [];

  let images = await getFiles("../images");

  for (let i = 0; i < images.length; i++) {
    const image = await Jimp.read(images[i]);

    const data = {
      res: image.getWidth() * image.getHeight(),
      image: `${images[i]} 4 5 10 15 20`,
    };
    const contain = resolutions.find((value) => value.res === data.res);
    if (!contain) {
      resolutions.push(data);
    }
  }

  resolutions.sort((a, b) => b.res - a.res);

  const fileName = "ranking-diff-images.json";
  const file = fs.createWriteStream(fileName);

  const step = resolutions.length / 20;
  for (let i = 0; i < resolutions.length; i += step) {
    const parse = JSON.stringify(resolutions[Math.round(i)]);
    file.write(`${parse}\n`);
  }

  file.close();
}

compareImage();
