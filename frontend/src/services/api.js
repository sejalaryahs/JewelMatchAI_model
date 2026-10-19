// const API_BASE_URL = "http://127.0.0.1:5000";

// export async function findMatches(imageFile, source, topK = 5) {
//   const formData = new FormData();

//   formData.append("image", imageFile);
//   formData.append("source", source);
//   formData.append("top_k", String(topK));

//   const response = await fetch(`${API_BASE_URL}/api/match`, {
//     method: "POST",
//     body: formData,
//   });

//   const data = await response.json();

//   if (!response.ok || !data.success) {
//     throw new Error(data.error || "Failed to find jewellery matches.");
//   }

//   return data;
// }

// export function getCatalogueImageUrl(imageGridFsId) {
//   if (!imageGridFsId) {
//     return "";
//   }

//   return `${API_BASE_URL}/catalogue-image/stored/${encodeURIComponent(
//     imageGridFsId,
//   )}`;
// }
// ##################################################################
// ###############################################################################

const API_BASE_URL = "http://127.0.0.1:5000";

/**
 * Find jewellery matches using the existing production matcher.
 *
 * This function preserves the existing /api/match endpoint.
 */
export async function findMatches(imageFile, source, topK = 5) {
  return requestMatches("/api/match", imageFile, source, topK);
}

/**
 * Find jewellery matches using the experimental DINOv2 model.
 *
 * The backend experimental endpoint will be added separately.
 * This function does not replace or modify the production matcher.
 */
export async function findExperimentalMatches(imageFile, source, topK = 5) {
  return requestMatches("/api/match/experimental", imageFile, source, topK);
}

/**
 * Shared request helper.
 */
async function requestMatches(endpoint, imageFile, source, topK) {
  if (!imageFile) {
    throw new Error("Please select a jewellery image.");
  }

  if (!["gold", "prototype"].includes(source)) {
    throw new Error("Source must be 'gold' or 'prototype'.");
  }

  const formData = new FormData();

  formData.append("image", imageFile);
  formData.append("source", source);
  formData.append("top_k", String(topK));

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method: "POST",
    body: formData,
  });

  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error(
      `The server returned an invalid response (${response.status}).`,
    );
  }

  if (!response.ok || !data.success) {
    throw new Error(data.error || "Failed to find jewellery matches.");
  }

  return data;
}

/**
 * Retrieve a catalogue image stored in MongoDB GridFS.
 */
export function getCatalogueImageUrl(imageGridFsId) {
  if (!imageGridFsId) {
    return "";
  }

  return `${API_BASE_URL}/catalogue-image/stored/${encodeURIComponent(
    imageGridFsId,
  )}`;
}
// ``` id="xk1m2p"
