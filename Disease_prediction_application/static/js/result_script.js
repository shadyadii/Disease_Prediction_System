const resultTitle = document.getElementById("result-title");
const resultDescription = document.getElementById("result-description");
const linkBack = document.getElementById("link-back");

// Example disease result based on user-provided symptoms
const disease = {
  title: "Common Cold",
  description: "The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Most people recover from a common cold in 7 to 10 days."
};

resultTitle.textContent = disease.title;
resultDescription.textContent = disease.description;
linkBack.href = "/templates/index.html"; // Replace with the path to your symptoms page