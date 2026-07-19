// Page-shell behaviour shared by the landing and the check page: navigation
// and scroll reveal. No check logic here — that lives in app.js on check.html.

(function () {
  const nav = document.querySelector("#nav");
  const burger = document.querySelector("#navBurger");

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
