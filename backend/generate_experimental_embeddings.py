import hashlib
import time

from gridfs import GridFS
from PIL import Image
from bson import ObjectId

from .database.mongodb import (
    get_database,
    get_gold_collection,
    get_prototype_collection,
)
from .experimental_matcher import ExperimentalJewelleryMatcher


def get_model_identifier(checkpoint_path):
    """Identify the exact experimental checkpoint."""
    digest = hashlib.sha256()

    with open(checkpoint_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return f"segmented_dinov2_experiment:{digest.hexdigest()[:16]}"


def main():
    started = time.perf_counter()

    matcher = ExperimentalJewelleryMatcher()
    database = get_database()
    gridfs = GridFS(database)

    model_identifier = get_model_identifier(matcher.checkpoint_path)

    print("\nExperimental model identifier:", model_identifier)
    print("Database:", database.name)
    print("Existing production embeddings will NOT be modified.")

    collections = {
        "gold": get_gold_collection(),
        "prototype": get_prototype_collection(),
    }

    for collection_name, collection in collections.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {collection_name.upper()} catalogue")
        print(f"{'=' * 60}")

        documents = list(collection.find({}))
        total = len(documents)
        saved = 0
        skipped = 0
        failed = 0

        print(f"Records found: {total}")

        for index, document in enumerate(documents, start=1):
            document_id = document["_id"]

            # Skip only if an embedding for this exact checkpoint
            # already exists.
            if (
                document.get("experimental_embedding_model") == model_identifier
                and isinstance(document.get("experimental_embedding"), list)
                and len(document["experimental_embedding"]) == 512
            ):
                skipped += 1
                print(
                    f"[{index}/{total}] Already embedded: "
                    f"{document.get('id', document_id)}"
                )
                continue

            image_id = document.get("image_gridfs_id")

            if image_id is None:
                print(f"[WARN] Missing image_gridfs_id: {document_id}")
                failed += 1
                continue

            try:
                if not isinstance(image_id, ObjectId):
                    image_id = ObjectId(str(image_id))

                with gridfs.get(image_id) as grid_file:
                    with Image.open(grid_file) as image:
                        embedding = matcher._embed_image(image)

                vector = embedding.astype("float32").tolist()

                if len(vector) != 512:
                    raise ValueError(f"Expected 512 dimensions, got {len(vector)}")

                # Update only experimental fields.
                result = collection.update_one(
                    {"_id": document_id},
                    {
                        "$set": {
                            "experimental_embedding": vector,
                            "experimental_embedding_model": model_identifier,
                        }
                    },
                )

                if result.matched_count == 1:
                    saved += 1
                    print(
                        f"[{index}/{total}] Saved: "
                        f"{document.get('id', document_id)}"
                    )
                else:
                    failed += 1
                    print(
                        f"[WARN] Record disappeared during processing: "
                        f"{document_id}"
                    )

            except Exception as exc:
                failed += 1
                print(f"[ERROR] {collection_name} record {document_id}: {exc}")

        print(f"\n{collection_name.upper()} summary")
        print(f"Saved:   {saved}")
        print(f"Skipped: {skipped}")
        print(f"Failed:  {failed}")

    print(f"\nCompleted in {time.perf_counter() - started:.2f} seconds.")
    print("Production embeddings were preserved.")


if __name__ == "__main__":
    main()
