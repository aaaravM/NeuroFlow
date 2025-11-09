document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tab');
  const forms = {
    signin: document.getElementById('signin'),
    signup: document.getElementById('signup'),
  };

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      Object.values(forms).forEach((f) => f.classList.remove('active'));
      forms[target].classList.add('active');
    });
  });

  forms.signin.addEventListener('submit', (e) => {
    e.preventDefault();
    const email = document.getElementById('signin-email').value.trim();
    // Simulated auth: store a basic session marker
    localStorage.setItem('nf_user', JSON.stringify({ email }));
    location.href = 'study.html';
  });

  forms.signup.addEventListener('submit', (e) => {
    e.preventDefault();
    const name = document.getElementById('signup-name').value.trim();
    const email = document.getElementById('signup-email').value.trim();
    localStorage.setItem('nf_user', JSON.stringify({ name, email }));
    location.href = 'study.html';
  });
});
