// Landing-shell behaviour only: navigation, scroll reveal, and the marketing
// CTAs that hand off to the verification console in app.js. No check logic here.

(function () {
  const nav = document.querySelector("#nav");
  const burger = document.querySelector("#navBurger");
  const startButton = document.querySelector("#startVerification");
  const demo = document.querySelector("#demo");

  // Hairline under the nav only once the page has moved, so the hero sits on an
  // uninterrupted background.
  const syncNavBorder = () => nav.classList.toggle("scrolled", window.scrollY > 8);
  syncNavBorder();
  window.addEventListener("scroll", syncNavBorder, { passive: true });

  burger.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    burger.setAttribute("aria-expanded", String(open));
  });

  nav.querySelectorAll(".nav-menu a").forEach((link) => {
    link.addEventListener("click", () => {
      nav.classList.remove("open");
      burger.setAttribute("aria-expanded", "false");
    });
  });

  // Every "пройти проверку" CTA scrolls to the console and starts a session, so
  // the landing page and the console never disagree about what the button does.
  document.querySelectorAll("[data-start-demo]").forEach((trigger) => {
    trigger.addEventListener("click", (event) => {
      event.preventDefault();
      nav.classList.remove("open");
      burger.setAttribute("aria-expanded", "false");
      demo.scrollIntoView({ behavior: "smooth", block: "start" });
      // Let the smooth scroll begin before the camera permission prompt appears.
      window.setTimeout(() => startButton.click(), 420);
    });
  });

  const reveals = document.querySelectorAll("[data-reveal]");
  if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    reveals.forEach((group) => group.classList.add("revealed"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("revealed");
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -12% 0px" },
  );

  reveals.forEach((group) => observer.observe(group));
})();
