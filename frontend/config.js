// Runtime config for the verification console. COPYed into the frontend image
// at build time: edit, then rebuild the frontend service to apply.
// Optional extras (see the appConfig reads at the top of app.js):
//   enableTestSkip: true — show the "skip step" test button.
window.APP_CONFIG = {
  mlApiUrl: "http://localhost:8100",
};
