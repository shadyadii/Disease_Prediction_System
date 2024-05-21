document.addEventListener("DOMContentLoaded", function() {
  const loginForm = document.getElementById("login");
  const createAccountForm = document.getElementById("createAccount");
  const linkCreateAccount = document.getElementById("linkCreateAccount");
  const linkLogin = document.getElementById("linkLogin");

  // Default credentials
  const defaultEmail = "tanmaya@gmail.com";
  const defaultPassword = "Password!123";

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

    // Get form values
    const email = loginForm.querySelector('input[name="email"]').value;
    const password = loginForm.querySelector('input[name="password"]').value;



    // Check if email and password match the default credentials
    if (email === defaultEmail && password === defaultPassword) {
      // If validation passes and credentials are correct, redirect to the home screen
     // window.location.href = "/index1.html";
    } else {
      alert("Invalid email or password. Please try again.");
    }
  });

  // Handle create account form submission
  createAccountForm.addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent default form submission behavior

    // Get form values
    const email = createAccountForm.querySelector('input[name="email"]').value;
    const password = createAccountForm.querySelector('input[name="password"]').value;

    // Perform form validation
    if (!validateEmail(email)) {
      alert("Please enter a valid email address.");
      return;
    }
    
    if (!validatePassword(password)) {
      alert("Password must be at least 8 characters long and contain at least one special character.");
      return;
    }
  });

  // Function to validate email address
  function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  // Function to validate password
  function validatePassword(password) {
    const passwordRegex = /^(?=.*[!@#$%^&*(),.?":{}|<>]).{8,}$/;
    return passwordRegex.test(password);
  }
});
