const tf = require("@tensorflow/tfjs-node");

const findSimilarImages = async (
  text,
  textEmbeddings,
  imageEmbeddings,
  useModel
) => {
  const textEmbedding = await useModel.embed([text]);

  // Normalize text embedding
  const normalizedTextEmbedding = textEmbedding.div(
    tf.norm(textEmbedding, 2, -1, true)
  );

  // Compute cosine similarity
  const similarities = tf.matMul(
    normalizedTextEmbedding,
    imageEmbeddings,
    false,
    true
  );
  const sortedIndices = similarities
    .squeeze()
    .arraySync()
    .map((sim, idx) => ({ sim, idx }))
    .sort((a, b) => b.sim - a.sim)
    .map((item) => item.idx);
  return sortedIndices;
};

module.exports = { findSimilarImages };
