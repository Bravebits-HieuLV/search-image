const path = require("path");
const { loadDataset } = require("./dataLoader");
const { embedImages, embedTexts } = require("./embedder");
const { findSimilarImages } = require("./search");
const mobilenet = require("@tensorflow-models/mobilenet");
const use = require("@tensorflow-models/universal-sentence-encoder");
const tf = require("@tensorflow/tfjs-node");

const main = async () => {
  const imageDir = path.join(__dirname, "../dataset/icons/");

  // Load image paths and corresponding captions
  const { imagePaths, textDescriptions } = await loadDataset(imageDir);

  // Load pre-trained models
  const mobilenetModel = await mobilenet.load();
  const useModel = await use.load();

  // Generate embeddings for images and text descriptions
  const imageEmbeddings = await embedImages(imagePaths, mobilenetModel);
  const textEmbeddings = await embedTexts(textDescriptions, useModel);

  // Search term
  const searchTerm = "house";

  // Find and print similar images
  const similarImagesIndices = await findSimilarImages(
    searchTerm,
    textEmbeddings,
    imageEmbeddings,
    useModel
  );
  console.log("Similar images for:", searchTerm);
  similarImagesIndices
    .slice(0, 5)
    .forEach((idx) => console.log(imagePaths[idx]));
};

main();
