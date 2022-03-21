import React, { useState } from "react";
// import loginImg from "../../login.svg";

const StoreLogin = (props) => {
	const [loginDetails, setLoginDetails] = useState({
		username: "",
		password: "",
	});
	const [loginSuccessful, setLoginSuccessful] = useState(false);
	const handleChange = (e) => {
		const input = e.target.name;
		const value = e.target.value;

		setLoginDetails({ ...loginDetails, [input]: value });
	};

	const handleSubmit = async (e) => {
		e.preventDefault();
		const formData = new FormData();

		formData.append("type", "Login");
		formData.append("username", loginDetails.username);
		formData.append("password", loginDetails.password);

		const response = await fetch("http://127.0.0.1:8000/storeowner", {
			method: "POST",
			body: formData,
		});
		const result = await response.json();

		if (result["loginStatus"] == true) {
			// setRegistered(true);
			// setLoginSuccessful(() => true);
			props.loginStatusDetails.changeLoginStatus(true);
			props.storeNum.changeStoreNum(result["storeNum"]);
			props.storeOwnerEmail.changeStoreOwnerEmail(result["storeOwnerEmail"]);
		} else {
			alert("Wrong credentials !");
		}
	};
	return (
		<div className="base-container" ref={props.containerRef}>
			<h1>
				<strong>Store</strong>
			</h1>
			<div className="header">Login</div>
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
							type="password"
							name="password"
							placeholder="password"
							onChange={handleChange}
						/>
					</div>
				</div>
			</div>
			<a href="" style={{ color: "black", textDecoration: "none" }}>
				Forget Password?
			</a>
			<div className="footer">
				<button
					type="button"
					className="loginRegisterbtn"
					onClick={handleSubmit}
				>
					Login
				</button>
			</div>
		</div>
	);
};
export default StoreLogin;
