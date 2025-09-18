# Fix Index/Docstore Issue - TODO List

## Steps to Complete:

- [ ] Step 1: Create missing directories
  - [ ] Create `data/raw/` directory for PDF files
  
- [ ] Step 2: Fix the error message in app.py
  - [ ] Update error message to show correct script paths (remove "scripts/" prefix)
  
- [ ] Step 3: Run the ingestion pipeline
  - [ ] Run `python ingest.py` to process any PDFs in `data/raw/`
  - [ ] Run `python chunk_and_index.py` to create the index and docstore
  
- [ ] Step 4: Verify the setup
  - [ ] Check that the artifacts are created properly
  - [ ] Test the app to ensure it works

## Progress:
- Starting implementation...
