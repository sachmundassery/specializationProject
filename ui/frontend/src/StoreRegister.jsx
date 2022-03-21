import React, { useState } from "react";
// import loginImg from "../../login.svg";

const StoreRegister = (props) => {
	const [registerDetails, setRegisterDetails] = useState({
		username: "",
		email: "",
		password: "",
		storenum: "",
	});

	const handleChange = (e) => {
		e.preventDefault();
		const input = e.target.name;

		const value = e.target.value;
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

		// var storenumRegex = /[1-1115]/;
		// console.log("-------", registerDetails.storenum);
		// if (!registerDetails.storenum.match(storenumRegex)) {
		// 	alert("input must be a number and we have only 1115 stores");
		// 	return false;
		// }
		return true;
	};

	const handleSubmit = async (e) => {
		e.preventDefault();

		if (handleValidation()) {
			const formData = new FormData();

			formData.append("type", "Register");
			formData.append("username", registerDetails.username);
			formData.append("email", registerDetails.email);
			formData.append("password", registerDetails.password);
			formData.append("storenum", registerDetails.storenum);
			console.log("Inside storeowner registration form before submition");

			const response = await fetch("http://127.0.0.1:8000/storeowner", {
				method: "POST",
				body: formData,
			});
			console.log("Inside storeowner registration form after submition");
			const result = await response.json();
			if (result["response"] == true) {
				alert("User Registered");
			}
		} else {
			console.log("false");
		}
	};

	return (
		<div className="base-container" ref={props.containerRef}>
			<h1>
				<strong>Store</strong>
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
							name="storenum"
							placeholder="store number"
							onChange={handleChange}
						/>
					</div>
					<div className="form-group">
						<input
							type="email"
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
	);
};
export default StoreRegister;
