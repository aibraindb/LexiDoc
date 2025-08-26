import { initializeApp, getApps, getApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';

const firebaseConfig = {
  "projectId": "intelibank",
  "appId": "1:388063551963:web:44309f9bcfc8cfc13607fe",
  "storageBucket": "intelibank.firebasestorage.app",
  "apiKey": "AIzaSyCd1rDBqIMnwbMFsC1CEs5t8JWZeC2K1Fc",
  "authDomain": "intelibank.firebaseapp.com",
  "measurementId": "",
  "messagingSenderId": "388063551963"
};

const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

export { app, auth, googleProvider };
