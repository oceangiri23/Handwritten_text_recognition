import React, { useEffect, useState } from "react";
import { IconButton, Button, Dialog, DialogActions, DialogContent, DialogTitle, TextField } from "@material-ui/core";
import { SendOutlined } from "@material-ui/icons";
import moment from "moment";
import { useAuthState } from "react-firebase-hooks/auth";
import { useHistory, useParams } from "react-router-dom";
import Announcement from "../components/Announcement";
import { auth, db, storage } from "../firebase"; // Ensure your firebase.js exports storage
import "./Class.css";

function Class() {
  const [classData, setClassData] = useState({});
  const [announcementContent, setAnnouncementContent] = useState("");
  const [assignmentContent, setAssignmentContent] = useState({
    title: "",
    description: "",
    image: null,
    score: "",
    deadline: "",
    base_text: ''
  });
  const [posts, setPosts] = useState([]);
  const [user, loading] = useAuthState(auth);
  const { id } = useParams();
  const history = useHistory();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const unsubscribe = db.collection("classes")
      .doc(id)
      .onSnapshot((snapshot) => {
        const data = snapshot.data();
        if (!data) {
          history.replace("/");
        } else {
          setClassData(data);
          setPosts(data.posts ? data.posts.reverse() : []);
        }
      });

    return () => unsubscribe();
  }, [id, history]);

  useEffect(() => {
    if (loading) return;
    if (!user) history.replace("/");
  }, [loading, user, history]);

  const createPost = async () => {
    if (!announcementContent) return;

    try {
      const myClassRef = db.collection("classes").doc(id);
      const myClassData = (await myClassRef.get()).data();

      let tempPosts = myClassData.posts || [];
      tempPosts.push({
        authorId: user.uid,
        content: announcementContent,
        date: moment().format("MMM Do YY"),
        image: user.photoURL,
        name: user.displayName,
      });

      await myClassRef.update({
        posts: tempPosts,
      });
      setAnnouncementContent(""); // Clear input after posting
    } catch (error) {
      console.error(error);
      alert("There was an error posting the announcement, please try again!");
    }
  };

  const createAssignment = async () => {
    try {
      const { title, description, score, deadline, image, base_text } = assignmentContent;

      let imageUrl = null; // Variable to store the image URL

      if (image) {
        const storageRef = storage.ref(`assignments/${image.name}`);
        await storageRef.put(image); // Upload the file
        imageUrl = await storageRef.getDownloadURL(); // Get the download URL
      }

      const myClassRef = db.collection("classes").doc(id);
      const myClassData = (await myClassRef.get()).data();

      const newAssignment = {
        title,
        description,
        score,
        deadline,
        imageUrl,
        base_text
      };

      const assignmentsArray = myClassData.assignments || []; // Ensure it exists
      assignmentsArray.push(newAssignment);

      await myClassRef.update({
        assignments: assignmentsArray,
      });

      // Reset input and close dialog
      setAssignmentContent({ title: "", description: "", image: null, score: "", deadline: "", base_text: '' });
      setOpen(false);
      alert("Assignment created successfully!");
    } catch (error) {
      console.error(error);
      alert(`There was an error creating the assignment: ${error.message}`);
    }
  };

  return (
    <div className="class">
      <div className="class__nameBox" style={{ position: 'relative', height: '200px', width: '100%' }}>
        <div className="class__name">{classData?.name}</div>

        <a
          href="/assignment"
          style={{
            position: 'absolute',
            bottom: '10px',
            right: '10px',
            textDecoration: 'none',
            color: 'white',
          }}
        >
         View Assignments
        </a>
      </div>

      <div className="class__announce">
        <img src={user?.photoURL} alt="User" />
        <input
          type="text"
          value={announcementContent}
          onChange={(e) => setAnnouncementContent(e.target.value)}
          placeholder="Announce something to your class"
        />
        <IconButton onClick={createPost}>
          <SendOutlined />
        </IconButton>
      </div>
      {classData.creatorUid === user?.uid && (
        <Button variant="contained" color="primary" onClick={() => setOpen(true)}>
          Add Assignment
        </Button>
      )}
      <Dialog open={open} onClose={() => setOpen(false)} aria-labelledby="form-dialog-title">
        <DialogTitle id="form-dialog-title">Create Assignment</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Title"
            type="text"
            fullWidth
            value={assignmentContent.title}
            onChange={(e) => setAssignmentContent({ ...assignmentContent, title: e.target.value })}
          />
          <TextField
            margin="dense"
            label="Description"
            type="text"
            fullWidth
            multiline
            rows={4}
            value={assignmentContent.description}
            onChange={(e) => setAssignmentContent({ ...assignmentContent, description: e.target.value })}
          />
          <TextField
            margin="dense"
            label="base_text"
            type="text"
            fullWidth
            multiline
            rows={4}
            value={assignmentContent.base_text}
            onChange={(e) => setAssignmentContent({ ...assignmentContent, base_text: e.target.value })}
          />
          <TextField
            margin="dense"
            label="Score"
            type="number"
            fullWidth
            value={assignmentContent.score}
            onChange={(e) => setAssignmentContent({ ...assignmentContent, score: e.target.value })}
          />
          <TextField
            margin="dense"
            label="Deadline"
            type="datetime-local"
            fullWidth
            InputLabelProps={{ shrink: true }}
            value={assignmentContent.deadline}
            onChange={(e) => setAssignmentContent({ ...assignmentContent, deadline: e.target.value })}
          />
          <input
            type="file"
            id="assignmentImage"
            onChange={(e) => setAssignmentContent({ ...assignmentContent, image: e.target.files[0] })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpen(false)} color="primary">
            Cancel
          </Button>
          <Button onClick={createAssignment} color="primary">
            Create Assignment
          </Button>
        </DialogActions>
      </Dialog>
      {posts?.map((post, index) => (
        <Announcement
          key={index} // Add key prop for unique identification
          authorId={post.authorId}
          content={post.content}
          date={post.date}
          image={post.image}
          name={post.name}
        />
      ))}
    </div>
  );
}

export default Class;
