// import { useEffect, useState } from "react";

// import { findMatches, getCatalogueImageUrl } from "./services/api";

// import "./App.css";

// function App() {
//   const [source, setSource] = useState("prototype");

//   const [selectedFile, setSelectedFile] = useState(null);

//   const [previewUrl, setPreviewUrl] = useState("");

//   const [results, setResults] = useState([]);

//   const [loading, setLoading] = useState(false);

//   const [error, setError] = useState("");

//   const [searched, setSearched] = useState(false);

//   // --------------------------------------------------------
//   // CLEAN PREVIEW URL
//   // --------------------------------------------------------

//   useEffect(() => {
//     return () => {
//       if (previewUrl) {
//         URL.revokeObjectURL(previewUrl);
//       }
//     };
//   }, [previewUrl]);

//   // --------------------------------------------------------
//   // FILE SELECT
//   // --------------------------------------------------------

//   function handleFileChange(event) {
//     const file = event.target.files?.[0];

//     if (!file) {
//       return;
//     }

//     if (!file.type.startsWith("image/")) {
//       setError("Please select an image file.");

//       return;
//     }

//     setError("");

//     setResults([]);

//     setSearched(false);

//     setSelectedFile(file);

//     const url = URL.createObjectURL(file);

//     setPreviewUrl(url);
//   }

//   // --------------------------------------------------------
//   // SOURCE CHANGE
//   // --------------------------------------------------------

//   function handleSourceChange(newSource) {
//     setSource(newSource);

//     setResults([]);

//     setSearched(false);

//     setError("");
//   }

//   // --------------------------------------------------------
//   // MATCH
//   // --------------------------------------------------------

//   async function handleFindMatches() {
//     if (!selectedFile) {
//       setError("Please upload a jewellery image first.");

//       return;
//     }

//     setLoading(true);

//     setError("");

//     setResults([]);

//     setSearched(false);

//     try {
//       const data = await findMatches(selectedFile, source, 5);

//       setResults(data.results || []);

//       setSearched(true);
//     } catch (err) {
//       console.error(err);

//       setError(err.message || "Unable to find matches.");
//     } finally {
//       setLoading(false);
//     }
//   }

//   // --------------------------------------------------------
//   // RESET
//   // --------------------------------------------------------

//   function handleReset() {
//     setSelectedFile(null);

//     setPreviewUrl("");

//     setResults([]);

//     setError("");

//     setSearched(false);

//     const input = document.getElementById("jewellery-upload");

//     if (input) {
//       input.value = "";
//     }
//   }

//   const targetName = source === "prototype" ? "Gold" : "Prototype";

//   return (
//     <div className="app">
//       <header className="header">
//         <div>
//           <h1>Jewellery Matcher</h1>

//           <p>Find visually similar jewellery using AI</p>
//         </div>
//       </header>

//       <main className="container">
//         {/* ------------------------------------------------ */}
//         {/* SOURCE SELECTION                                 */}
//         {/* ------------------------------------------------ */}

//         <section className="source-section">
//           <h2>Search From</h2>

//           <div className="source-buttons">
//             <button
//               type="button"
//               className={
//                 source === "gold" ? "source-button active" : "source-button"
//               }
//               onClick={() => handleSourceChange("gold")}
//             >
//               Gold
//             </button>

//             <button
//               type="button"
//               className={
//                 source === "prototype"
//                   ? "source-button active"
//                   : "source-button"
//               }
//               onClick={() => handleSourceChange("prototype")}
//             >
//               Prototype
//             </button>
//           </div>

//           <p className="direction-text">
//             {source === "prototype" ? "Prototype → Gold" : "Gold → Prototype"}
//           </p>
//         </section>

//         {/* ------------------------------------------------ */}
//         {/* UPLOAD                                       */}
//         {/* ------------------------------------------------ */}

//         <section className="upload-section">
//           <h2>Upload Jewellery Image</h2>

//           <label htmlFor="jewellery-upload" className="upload-box">
//             {!previewUrl && (
//               <div className="upload-placeholder">
//                 <div className="upload-icon">+</div>

//                 <strong>Choose Jewellery Image</strong>

//                 <span>JPG, JPEG, PNG or WEBP</span>
//               </div>
//             )}

//             {previewUrl && (
//               <img
//                 src={previewUrl}
//                 alt="Uploaded jewellery"
//                 className="preview-image"
//               />
//             )}
//           </label>

//           <input
//             id="jewellery-upload"
//             type="file"
//             accept="image/*"
//             onChange={handleFileChange}
//             hidden
//           />

//           {selectedFile && (
//             <div className="selected-file">
//               <span>{selectedFile.name}</span>

//               <button type="button" onClick={handleReset}>
//                 Remove
//               </button>
//             </div>
//           )}

//           <button
//             type="button"
//             className="match-button"
//             onClick={handleFindMatches}
//             disabled={!selectedFile || loading}
//           >
//             {loading ? "Finding Matches..." : "Find Matches"}
//           </button>
//         </section>

//         {/* ------------------------------------------------ */}
//         {/* ERROR                                         */}
//         {/* ------------------------------------------------ */}

//         {error && <div className="error-box">{error}</div>}

//         {/* ------------------------------------------------ */}
//         {/* RESULTS                                       */}
//         {/* ------------------------------------------------ */}

//         {searched && (
//           <section className="results-section">
//             <div className="results-heading">
//               <div>
//                 <h2>Matching {targetName} Jewellery</h2>

//                 <p>Top {results.length} visual matches</p>
//               </div>
//             </div>

//             {results.length === 0 && (
//               <div className="no-results">No matching jewellery was found.</div>
//             )}

//             {results.length > 0 && (
//               <div className="results-grid">
//                 {results.map((result, index) => {
//                   const imageDomain = result.domain;

//                   return (
//                     <article
//                       className={
//                         index === 0 ? "result-card best-match" : "result-card"
//                       }
//                       key={`${result.id}-${index}`}
//                     >
//                       {index === 0 && (
//                         <div className="best-badge">Best Match</div>
//                       )}

//                       <div className="result-image-wrapper">
//                         <img
//                           src={getCatalogueImageUrl(result.image_gridfs_id)}
//                           alt={result.id}
//                           className="result-image"
//                         />
//                       </div>

//                       <div className="result-info">
//                         <h3>{result.id}</h3>

//                         <div className="similarity">
//                           <span>Similarity</span>

//                           <strong>{result.similarity_percent}%</strong>
//                         </div>
//                       </div>
//                     </article>
//                   );
//                 })}
//               </div>
//             )}
//           </section>
//         )}
//       </main>
//     </div>
//   );
// }

// export default App;
// ###########################
import { useEffect, useState } from "react";

import {
  findMatches,
  findExperimentalMatches,
  getCatalogueImageUrl,
} from "./services/api";

import "./App.css";

function App() {
  const [source, setSource] = useState("prototype");
  const [modelType, setModelType] = useState("production");

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");

  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);

  // Release the previous preview URL when it changes.
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function handleFileChange(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setError("");
    setResults([]);
    setSearched(false);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }

  function handleSourceChange(newSource) {
    setSource(newSource);
    setResults([]);
    setSearched(false);
    setError("");
  }

  function handleModelChange(newModel) {
    setModelType(newModel);
    setResults([]);
    setSearched(false);
    setError("");
  }

  async function handleFindMatches() {
    if (!selectedFile) {
      setError("Please upload a jewellery image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResults([]);
    setSearched(false);

    try {
      const matchFunction =
        modelType === "experimental" ? findExperimentalMatches : findMatches;

      const data = await matchFunction(selectedFile, source, 5);

      setResults(data.results || []);
      setSearched(true);
    } catch (err) {
      console.error("Jewellery matching error:", err);
      setError(err.message || "Unable to find matches.");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setSelectedFile(null);
    setPreviewUrl("");
    setResults([]);
    setError("");
    setSearched(false);

    const input = document.getElementById("jewellery-upload");

    if (input) input.value = "";
  }

  const targetName = source === "prototype" ? "Gold" : "Prototype";

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Jewellery Matcher</h1>
          <p>Compare jewellery matching models using MongoDB catalogue data</p>
        </div>
      </header>

      <main className="container">
        <section className="source-section">
          <h2>Search From</h2>

          <div className="source-buttons">
            <button
              type="button"
              className={
                source === "gold" ? "source-button active" : "source-button"
              }
              onClick={() => handleSourceChange("gold")}
            >
              Gold
            </button>

            <button
              type="button"
              className={
                source === "prototype"
                  ? "source-button active"
                  : "source-button"
              }
              onClick={() => handleSourceChange("prototype")}
            >
              Prototype
            </button>
          </div>

          <p className="direction-text">
            {source === "prototype" ? "Prototype → Gold" : "Gold → Prototype"}
          </p>
        </section>

        <section className="source-section">
          <h2>Select Model</h2>

          <div className="source-buttons">
            <button
              type="button"
              className={
                modelType === "production"
                  ? "source-button active"
                  : "source-button"
              }
              onClick={() => handleModelChange("production")}
            >
              Existing ONNX Model
            </button>

            <button
              type="button"
              className={
                modelType === "experimental"
                  ? "source-button active"
                  : "source-button"
              }
              onClick={() => handleModelChange("experimental")}
            >
              Experimental DINOv2
            </button>
          </div>

          <p className="direction-text">
            {modelType === "production"
              ? "Uses the existing matcher and stored production embeddings."
              : "Uses the experimental checkpoint with catalogue images from MongoDB."}
          </p>
        </section>

        <section className="upload-section">
          <h2>Upload Jewellery Image</h2>

          <label htmlFor="jewellery-upload" className="upload-box">
            {!previewUrl && (
              <div className="upload-placeholder">
                <div className="upload-icon">+</div>
                <strong>Choose Jewellery Image</strong>
                <span>JPG, JPEG, PNG or WEBP</span>
              </div>
            )}

            {previewUrl && (
              <img
                src={previewUrl}
                alt="Uploaded jewellery"
                className="preview-image"
              />
            )}
          </label>

          <input
            id="jewellery-upload"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileChange}
            hidden
          />

          {selectedFile && (
            <div className="selected-file">
              <span>{selectedFile.name}</span>

              <button type="button" onClick={handleReset}>
                Remove
              </button>
            </div>
          )}

          <button
            type="button"
            className="match-button"
            onClick={handleFindMatches}
            disabled={!selectedFile || loading}
          >
            {loading
              ? "Finding Matches..."
              : modelType === "experimental"
                ? "Test Experimental Model"
                : "Find Matches"}
          </button>
        </section>

        {error && <div className="error-box">{error}</div>}

        {searched && (
          <section className="results-section">
            <div className="results-heading">
              <div>
                <h2>{targetName} Matching Results</h2>
                <p>
                  {modelType === "experimental"
                    ? "Experimental DINOv2"
                    : "Existing ONNX Model"}{" "}
                  · Top {results.length} matches
                </p>
              </div>
            </div>

            {results.length === 0 && (
              <div className="no-results">
                No matching jewellery was found in the target catalogue.
              </div>
            )}

            {results.length > 0 && (
              <div className="results-grid">
                {results.map((result, index) => (
                  <article
                    className={
                      index === 0 ? "result-card best-match" : "result-card"
                    }
                    key={`${result.mongo_id || result.id}-${index}`}
                  >
                    {index === 0 && (
                      <div className="best-badge">Top Ranked Match</div>
                    )}

                    <div className="result-image-wrapper">
                      <img
                        src={getCatalogueImageUrl(result.image_gridfs_id)}
                        alt={result.id || result.filename || "Jewellery match"}
                        className="result-image"
                        onError={(event) => {
                          event.currentTarget.style.visibility = "hidden";
                        }}
                      />
                    </div>

                    <div className="result-info">
                      <h3>{result.id || result.filename || "Jewellery"}</h3>

                      <div className="similarity">
                        <span>Similarity</span>
                        <strong>
                          {Number(result.similarity_percent ?? 0).toFixed(2)}%
                        </strong>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
// ``` id="app-exp"
