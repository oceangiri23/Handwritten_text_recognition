import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { auth, db, storage } from "../firebase";
import './AssignmentDetail.css';
import Dropzone from 'react-dropzone';
import { useAuthState } from 'react-firebase-hooks/auth';
import axios from 'axios'
const AssignmentDetail = () => {
    const { id } = useParams();
    const [submission, setSubmission] = useState({
        image_uploaded: null,
        text_obtained_from_calling: '',
    });
    const [user, loading, error] = useAuthState(auth);
    const [assignment, setAssignment] = React.useState(null);
    const [imageUrl, setImageUrl] = React.useState(null);

    const fetchAssignments = async () => {
        if (!user) return;
        try {
            const userDoc = await db.collection("users").where("uid", "==", user.uid).get();
            const enrolledClasses = userDoc.docs[0]?.data()?.enrolledClassrooms || [];
            if (!Array.isArray(enrolledClasses) || enrolledClasses.length === 0) {
                console.warn("No enrolled classes found or enrolledClasses is not an array.");
                return;
            }

            const assignmentsPromises = enrolledClasses.map(async (singleClass) => {
                const classDoc = await db.collection("classes").doc(singleClass.id).get();
                if (classDoc.exists) {
                    const classData = classDoc.data();
                    return classData.assignments || [];
                } else {
                    console.error("Class not found:", singleClass);
                    return [];
                }
            });

            const assignmentsArray = await Promise.all(assignmentsPromises);
            const allAssignments = assignmentsArray.flat();
            setAssignment(allAssignments.find(dat => dat.deadline === id)); // Use find instead of filter for single assignment
        } catch (error) {
            console.error("Error fetching assignments:", error);
        }
    };

    React.useEffect(() => {
        fetchAssignments();
    }, [user]);

    const OCR = async (image_url, base_text) => {
        console.log(image_url)
        try {
            const response = await axios.post("http://localhost:8000/upload-image", {
                image_url
            })
            console.log(response.data.prediction)
            const obtained_text = response.data.prediction;
            return obtained_text;
        } catch (error) {
            console.log(error)
        }
    };
    const graderApi = async (base_text) => {
        try {
            const response = await axios.post("http://localhost:8000/grade", { base_text, obtained_text: submission.text_obtained_from_calling });
            console.log(response.data)
            return response.data;
        }
        catch (e) {
            console.log(e);
        }
    }

    const uploadImage = async (image) => {
        const storageRef = storage.ref(`assignments/${image.name}`);
        await storageRef.put(image);
        const images = await storageRef.getDownloadURL();
        setImageUrl(images);
        const obtained_text = await OCR(images);
        setSubmission({
            image_uploaded: images,
            text_obtained_from_calling: obtained_text
        });
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!user) return;

        try {
            if (imageUrl === null) {
                throw new Error("Image not uploaded");

            }
            if (submission.text_obtained_from_calling === '') {
                throw new Error("Text not obtained from image");
            }
            if (assignment?.base_text === '') {
                throw new Error("Base text not found");
            }
            const score_ = await graderApi(assignment?.base_text || '');
            const newSubmission = {
                assignment: id,
                base_text: assignment?.base_text || '',
                obtained_text: submission.text_obtained_from_calling,
                score:parseInt(assignment.score)*score_,
                imageUrl
            };
            const userRef = await db.collection("users").where("uid", "==", user.uid).get();
            const docId = userRef.docs[0].id;
            const userData = userRef.docs[0].data();
            let userSubmission = userData.submissions || [];
            userSubmission.push(newSubmission);
            const docRef = await db.collection("users").doc(docId);
            await docRef.update({ submissions: userSubmission });
            alert("Assignment submitted successfully");
        } catch (e) {
            console.log(e);
        }
    };

    if (!assignment) {
        return <div>Loading...</div>; // Show loading indicator while assignment is being fetched
    }

    return (
        <div className="assignment-detail-container">
            <h2>{assignment?.title}</h2>
            <img src={assignment?.imageUrl} alt={assignment?.title} className="assignment-image" />
            <p><strong>Description:</strong> {assignment.description}</p>
            <p><strong>Deadline:</strong> {new Date(assignment?.deadline).toLocaleString()}</p>
            <p><strong>Score:</strong> {assignment.score}</p>

            <h3>Submit Your Assignment</h3>
            <form onSubmit={handleSubmit}>
                <Dropzone
                    onDrop={(acceptedFiles) => uploadImage(acceptedFiles[0])}
                    accept="image/*"
                >
                    {({ getRootProps, getInputProps }) => (
                        <div {...getRootProps({ className: 'dropzone' })}>
                            <input {...getInputProps()} />
                            <p>Drag 'n' drop an image here, or click to select one</p>
                        </div>
                    )}
                </Dropzone>
                <p>Image uploaded: {submission.image_uploaded ? 'Yes' : 'No'}</p>
                <img src={submission?.image_uploaded} alt={assignment?.title} className="assignment-image" />
                <p>Text obtained from image: {submission.text_obtained_from_calling}</p>
                {
                    submission.text_obtained_from_calling && <button type='submit' onClick={graderApi} style={{
                        backgroundColor: "green",
                        color: "white",
                        padding: "10px",
                        margin: "10px"
                    }}> Grade</button>
                }
            </form>
        </div>
    );
};

export default AssignmentDetail;
