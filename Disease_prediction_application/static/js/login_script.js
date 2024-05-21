document.addEventListener("DOMContentLoaded", function() {
    const loginForm = document.getElementById("login");
    const createAccountForm = document.getElementById("createAccount");
    const linkCreateAccount = document.getElementById("linkCreateAccount");
    const linkLogin = document.getElementById("linkLogin");
  
    // Show create account form when "Create account" link is clicked
    linkCreateAccount.addEventListener("click", function(event) {
      event.preventDefault();
      loginForm.classList.add("form--hidden");
      createAccountForm.classList.remove("form--hidden");
    });
  
    // Show login form when "Sign in" link is clicked
    linkLogin.addEventListener("click", function(event) {
      event.preventDefault();
      createAccountForm.classList.add("form--hidden");
      loginForm.classList.remove("form--hidden");
    });
  
    // Handle login form submission
    loginForm.addEventListener("submit", function(event) {
      event.preventDefault(); // Prevent default form submission behavior
  
      // Perform any necessary form validation here
      
      // If validation passes, you can redirect to the home screen
      window.location.href = "/templates/index.html";
    });
  
    // Handle create account form submission
    createAccountForm.addEventListener("submit", function(event) {
      event.preventDefault(); // Prevent default form submission behavior
  
      // Perform any necessary form validation here
      
      // If validation passes, you can redirect to the home screen
      window.location.href = "/templates/index.html";
    });
  });
  