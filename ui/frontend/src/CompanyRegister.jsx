import React, { useState } from "react";
import { ListGroupItem } from "react-bootstrap";
import CompanyLogin from "./CompanyLogin";
// import loginImg from "../../login.svg";
const CompanyRegister = (props) => {
	const [registerDetails, setRegisterDetails] = useState({
		username: "",
		email: "",
		password: "",
		companyname: "",
		companyidcard: "",
	});
	// const [registered, setRegistered] = useState(false)
	const handleChange = (e) => {
		e.preventDefault();
		const input = e.target.name;
		var value;
		if (e.target.name === "companyidcard") {
			value = e.target.files[0];
		} else {
			value = e.target.value;
		}

		setRegisterDetails({ ...registerDetails, [input]: value });
	};

	const handleValidation = () => {
		var usernameRegex = /^[a-zA-Z\-0-9_]+$/;
		if (!registerDetails.username.match(usernameRegex)) {
			alert("Username should not contain numbers, or special characters");
			return false;
		}

		var emailRegex =
			/[a-zA-Z0-9_]+[\.]?([a-zA-Z0-9]+)?[\@][a-z]{3,9}[\.][a-z]{2,5}/;
		if (!registerDetails.email.match(emailRegex)) {
			alert("email cannot contain special characters other thand @ and _");
			return false;
		}

		var passwordRegex =
			/^(?=.*[0-9])(?=.*[!@#$%^&*])[a-zA-Z0-9!@#$%^&*]{7,15}$/;
		if (!registerDetails.password.match(passwordRegex)) {
			alert(
				"password must be of 7 to 15 characters which contain at least one numeric digit and a special character"
			);
			return false;
		}

		if (registerDetails.companyidcard.length == 0) {
			alert("you need to upload the image of your company id card");
			return false;
		}
		return true;
	};

	const handleSubmit = async (e) => {
		e.preventDefault();

		if (handleValidation()) {
			const formData = new FormData();

			formData.append("type", "Register");
			formData.append("username", registerDetails.username);
			formData.append("companyname", registerDetails.companyname);
			formData.append("email", registerDetails.email);
			formData.append("password", registerDetails.password);
			formData.append("companyidcard", registerDetails.companyidcard);

			const response = await fetch("http://127.0.0.1:8000/company", {
				method: "POST",
				body: formData,
			});
			const result = await response.json();
			if (result["response"] == true) {
				alert("User Registered");
			}
		} else {
			console.log("false");
		}
	};

	return (
		<>
			<div className="base-container" ref={props.containerRef}>
				<h1>
					<strong>Company</strong>
				</h1>
				<div className="header">Register</div>
				<div className="content">
					<div className="image">{/* <img src={loginImg} /> */}</div>
					<div className="form">
						<div className="form-group">
							<input
								type="text"
								name="username"
								placeholder="username"
								onChange={handleChange}
							/>
						</div>
						<div className="form-group">
							<input
								type="text"
								name="companyname"
								placeholder="company name"
								onChange={handleChange}
							/>
						</div>
						<div className="form-group">
							<input
								type="text"
								name="email"
								placeholder="email"
								onChange={handleChange}
							/>
						</div>
						<div className="form-group">
							<input
								type="password"
								name="password"
								placeholder="password"
								onChange={handleChange}
							/>
						</div>
						<div className="form-group">
							<h6>Upload your company ID card.</h6>
							<input
								type="file"
								name="companyidcard"
								placeholder="Upload your company id"
								accept="image/*"
								onChange={handleChange}
							/>
						</div>
					</div>
				</div>
				<div className="footer">
					<button
						type="button"
						className="loginRegisterbtn"
						onClick={handleSubmit}
					>
						Register
					</button>
				</div>
			</div>
		</>
	);
};

export default CompanyRegister;
