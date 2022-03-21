import React from "react";
import Company from "./Company";

import CompanyLogin from "./CompanyLogin";
import CompanyRegister from "./CompanyRegister";

class CompanyLoginRegister extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			isLogginActive: true,
			loginStatus: false,
			userLoggedin: "",
		};
	}
	changeLoginStatus(state) {
		this.setState({ loginStatus: state });
	}
	changeUserLoggedin(userName) {
		console.log("companyloginregister.jsx", userName);
		this.setState({ userLoggedin: userName });
	}

	componentDidMount() {
		//Add .right by default
		this.rightSide.classList.add("right");
	}

	changeState() {
		const { isLogginActive } = this.state;

		if (isLogginActive) {
			this.rightSide.classList.remove("right");
			this.rightSide.classList.add("left");
		} else {
			this.rightSide.classList.remove("left");
			this.rightSide.classList.add("right");
		}
		this.setState((prevState) => ({
			isLogginActive: !prevState.isLogginActive,
		}));
	}

	render() {
		const { isLogginActive } = this.state;
		const current = isLogginActive ? "Register" : "Login";
		const currentActive = isLogginActive ? "login" : "register";

		return (
			<>
				{!this.state.loginStatus ? (
					<div className="LoginRegister">
						<div className="login">
							<div className="container" ref={(ref) => (this.container = ref)}>
								{isLogginActive && (
									<CompanyLogin
										containerRef={(ref) => (this.current = ref)}
										loginStatusDetails={{
											changeLoginStatus: this.changeLoginStatus.bind(this),
										}}
										userNameLoggedin={{
											changeUserLoggedin: this.changeUserLoggedin.bind(this),
										}}
									/>
								)}
								{!isLogginActive && (
									<CompanyRegister
										containerRef={(ref) => (this.current = ref)}
									/>
								)}
							</div>
							<RightSide
								current={current}
								currentActive={currentActive}
								containerRef={(ref) => (this.rightSide = ref)}
								onClick={this.changeState.bind(this)}
							/>
						</div>
					</div>
				) : (
					<Company userLoggedin={this.state.userLoggedin} />
				)}
			</>
		);
	}
}

const RightSide = (props) => {
	return (
		<div
			className="right-side"
			ref={props.containerRef}
			onClick={props.onClick}
		>
			<div className="inner-container">
				<div className="text">{props.current}</div>
			</div>
		</div>
	);
};

export default CompanyLoginRegister;
