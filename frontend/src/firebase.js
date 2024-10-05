import firebase from "firebase";

const firebaseConfig = {
  apiKey: "AIzaSyCyD1j9uM5pVCxyCndBnzbrMUvRItNqfTc",
  authDomain: "classroom-b1d4b.firebaseapp.com",
  projectId: "classroom-b1d4b",
  storageBucket: "classroom-b1d4b.appspot.com",
  messagingSenderId: "338650819571",
  appId: "1:338650819571:web:0b85e9801b97dfdac14086",
  measurementId: "G-K3TJ1TL7FN"
};

const app = firebase.initializeApp(firebaseConfig);
const auth = app.auth();
const db = app.firestore();
const googleProvider = new firebase.auth.GoogleAuthProvider();
const storage = app.storage();

// Sign in and check or create account in firestore
const signInWithGoogle = async () => {
  try {
    const response = await auth.signInWithPopup(googleProvider);
    console.log(response);
    console.log(response.user);
    const user = response.user;
    console.log(`User ID - ${user.uid}`);
    const querySnapshot = await db
      .collection("users")
      .where("uid", "==", user.uid)
      .get();
    if (querySnapshot.docs.length === 0) {
      // create a new user
      await db.collection("users").add({
        uid: user.uid,
        enrolledClassrooms: [],
      });
    }
  } catch (err) {
    alert(err.message);
  }
};
const logout = () => {
  auth.signOut();
};

export { app, auth, db,storage, signInWithGoogle, logout };
