import React from 'react';
import './Assignment.css'; // Import the CSS file
import { useHistory } from 'react-router-dom';
import { auth, db } from "../firebase"; // Ensure your firebase.js exports db
import { useAuthState } from "react-firebase-hooks/auth";

// Assignment component
const Assignment = () => {
    const history = useHistory();
    const [user, loading, error] = useAuthState(auth);
    const [assignments, setAssignments] = React.useState([]);

    // Function to fetch assignments
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
                // Fetch class document
                const classDoc = await db.collection("classes").doc(singleClass.id).get();
                if (classDoc.exists) {
                    const classData = classDoc.data();
                    return classData.assignments || []; // Return the assignments or an empty array
                } else {
                    console.error("Class not found:", singleClass);
                    return [];
                }
            });

            // Wait for all assignments to be fetched
            const assignmentsArray = await Promise.all(assignmentsPromises);
            console.log(assignmentsArray)
            // Flatten the array of assignments
            const allAssignments = assignmentsArray.flat();
            console.log(allAssignments)
            // Update the state with all fetched assignments
            setAssignments(allAssignments);

         
        } catch (error) {
            console.error("Error fetching assignments:", error);
        }
    };

    // Fetch assignments when user state changes
    React.useEffect(() => {
        fetchAssignments();
    }, [user]);

    const goToAssignment = (id) => {
        history.push(`/assignment/${id}`);
    };

    return (
        <div className="assignment-container">
            <h2>Assignments</h2>
            <div className="assignment-list">
            {
                assignments.length === 0 && <p>No assignments found.</p>
            }
                {assignments.map((assignment) => (
                    <div key={assignment.deadline} className="assignment-card" onClick={() => goToAssignment(assignment.deadline)}>
                        <h3>{assignment.title}</h3>
                        <p>{assignment.description}</p>
                        <p className="due-date">Due Date: {assignment.dueDate}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Assignment;
