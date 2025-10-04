document.addEventListener("DOMContentLoaded", function () {
  const uploadBox = document.getElementById("uploadBox");
  const fileInput = document.getElementById("fileInput");
  const processingMessage = document.getElementById("processingMessage");
  const successMessage = document.getElementById("successMessage");
  const errorMessage = document.getElementById("errorMessage");

  if (uploadBox && fileInput) {
    // Click to upload
    uploadBox.addEventListener("click", () => {
      fileInput.click();
    });

    // Drag and drop functionality
    uploadBox.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadBox.classList.add("dragging");
    });

    uploadBox.addEventListener("dragleave", () => {
      uploadBox.classList.remove("dragging");
    });

    uploadBox.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadBox.classList.remove("dragging");

      const files = e.dataTransfer.files;
      if (files.length > 0) {
        fileInput.files = files;
        handleFileUpload(files[0]);
      }
    });

    // File input change
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  async function handleFileUpload(file) {
    // Validate file type
    const allowedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ];
    if (!allowedTypes.includes(file.type)) {
      showError("Invalid file type. Please upload a PDF or DOCX file.");
      return;
    }

    // Validate file size (16MB max)
    if (file.size > 16 * 1024 * 1024) {
      showError("File is too large. Maximum size is 16MB.");
      return;
    }

    // Hide messages
    errorMessage.style.display = "none";
    successMessage.style.display = "none";
    uploadBox.style.display = "none";
    processingMessage.style.display = "block";

    // Create form data
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        processingMessage.style.display = "none";
        successMessage.style.display = "block";
      } else {
        showError(data.error || "Failed to upload file");
      }
    } catch (error) {
      showError("Error uploading file: " + error.message);
    }
  }

  function showError(message) {
    processingMessage.style.display = "none";
    uploadBox.style.display = "block";
    errorMessage.textContent = message;
    errorMessage.style.display = "block";

    // Reset file input
    fileInput.value = "";
  }
});
