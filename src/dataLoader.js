const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const loadDataset = async (imageDir) => {
  const jsonPath = path.join(imageDir, "../icons_data.json");
  const data = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));

  const imagePaths = [];
  const textDescriptions = [];

  for (const item of data) {
    let imgPath = path.join(imageDir, item.image_path);
    if (path.extname(imgPath).toLowerCase() === ".svg") {
      const pngPath = imgPath.replace(".svg", ".png");
      await sharp(imgPath).png().toFile(pngPath);
      imgPath = pngPath;
    }
    imagePaths.push(imgPath);
    textDescriptions.push(item.captions);
  }

  return { imagePaths, textDescriptions };
};

module.exports = { loadDataset };
