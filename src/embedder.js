const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

const embedImages = async (imagePaths, mobilenetModel) => {
  const imageTensors = await Promise.all(
    imagePaths.map(async (imgPath) => {
      const imgBuffer = fs.readFileSync(imgPath);
      const imgTensor = tf.node.decodeImage(new Uint8Array(imgBuffer), 3);
      const processedImg = tf.image
        .resizeBilinear(imgTensor, [224, 224])
        .div(tf.scalar(255))
        .expandDims();

      // Use the 'conv_pw_13_relu' layer of MobileNet to get image embeddings
      const embedding = mobilenetModel
        .infer(processedImg, { pooling: "avg" })
        .squeeze();
      return embedding;
    })
  );
  const imageEmbeddings = tf.stack(imageTensors);

  // Project image embeddings to 512 dimensions using a dense layer
  const projectionLayer = tf.layers.dense({ units: 512, activation: "linear" });
  const projectedEmbeddings = projectionLayer.apply(imageEmbeddings);

  // Normalize the projected image embeddings
  return projectedEmbeddings.div(tf.norm(projectedEmbeddings, 2, -1, true));
};

const embedTexts = async (texts, useModel) => {
  const textEmbeddings = await useModel.embed(texts);

  // Normalize text embeddings
  return textEmbeddings.div(tf.norm(textEmbeddings, 2, -1, true));
};

module.exports = { embedImages, embedTexts };
