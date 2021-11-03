const fsPromises = require("fs").promises;
var fs = require("fs");
const Jimp = require("jimp");

async function getFiles(diretorio, arquivos) {
  if (!arquivos) arquivos = [];

  let listaDeArquivos = await fsPromises.readdir(diretorio);
  for (let k in listaDeArquivos) {
    let stat = await fsPromises.stat(diretorio + "/" + listaDeArquivos[k]);
    if (stat.isDirectory())
      await getFiles(diretorio + "/" + listaDeArquivos[k], arquivos);
    else arquivos.push(diretorio + "/" + listaDeArquivos[k]);
  }

  return arquivos;
}

async function compareImage() {
  const dataComparativeImages = [];

  let images = await getFiles("../images");

  for (let i = 0; i < images.length; i++) {
    const imageComparedTo = await Jimp.read(images[i]);
    console.log(`====================${images[i]}====================\n`);

    for (let j = i; j < images.length; j++) {
      const image = await Jimp.read(images[j]);

      if (i != j) {
        console.log(`Compare: [${images[i]}, ${images[j]}\n`);

        const data = {
          id: `${i}${j}`,
          Comparation: `(${images[i]} | ${images[j]})`,
          distance: Jimp.distance(imageComparedTo, image), ///distance: a distância de Hamming entre os hashes de duas imagens, ou seja. o número de bits que diferem.
          diffInPercent: Jimp.diff(imageComparedTo, image).percent, // diff: a diferença percentual entre duas imagens.
        };
        dataComparativeImages.push(data);
      }
      continue;
    }
  }

  const fileName = "ranking-diff-images.json";
  const file = fs.createWriteStream(fileName);

  dataComparativeImages.sort((a, b) => b.diffInPercent - a.diffInPercent);
  const parse = JSON.stringify(dataComparativeImages);

  file.write(`${parse}\n`);

  file.close();
}

compareImage();
