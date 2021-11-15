const fsPromises = require("fs").promises;
var fs = require("fs");
const Jimp = require("jimp");
var path = require("path");

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
  let resolutions = [];

  let images = await getFiles("../images");

  for (let i = 0; i < images.length; i++) {
    if (images[i] === "../images/image_teste_segementacao_3_classes.jpg") {
      continue;
    }
    const image = await Jimp.read(images[i]);

    const data = {
      res: image.getWidth() * image.getHeight(),
      image: `${path.basename(images[i])} 4 5 10 15 20`,
    };
    const contain = resolutions.find((value) => value.res === data.res);
    if (!contain) {
      resolutions.push(data);
    }
  }

  resolutions = resolutions.sort((a, b) => a.res - b.res);
  console.log(resolutions);

  const fileName = "experimental";
  const file = fs.createWriteStream(fileName);

  const step = Math.round(resolutions.length / 5);
  const picked = [];
  for (let i = 0; i < resolutions.length; i += step + 1) {
    picked.push(resolutions[i]);
    file.write(picked[picked.length - 1].image + "\n");
  }

  // file.write(path.basename(images[resolutions.length-1] + " 4 5 10 15 20\n"));
  // picked.push(resolutions[resolutions.length - 1]);

  file.close();
  console.log(picked);
}

compareImage();
