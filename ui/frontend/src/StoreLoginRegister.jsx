import React from "react";

import StoreLogin from "./StoreLogin";
import Storeowner from "./Storeowner";
import StoreRegister from "./StoreRegister";

class StoreLoginRegister extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			isLogginActive: true,
			loginStatus: false,
			storeNum: 0,
			storeOwnerEmail: "",
		};
	}
	changeLoginStatus(state) {
		this.setState({ loginStatus: state });
	}
	changeStoreNum(state) {
		this.setState({ storeNum: state });
	}
	changeStoreOwnerEmail(state) {
		this.setState({ storeOwnerEmail: state });
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
									<StoreLogin
										containerRef={(ref) => (this.current = ref)}
										loginStatusDetails={{
											changeLoginStatus: this.changeLoginStatus.bind(this),
										}}
										storeNum={{
											changeStoreNum: this.changeStoreNum.bind(this),
										}}
										storeOwnerEmail={{
											changeStoreOwnerEmail:
												this.changeStoreOwnerEmail.bind(this),
										}}
									/>
								)}
								{!isLogginActive && (
									<StoreRegister containerRef={(ref) => (this.current = ref)} />
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
					<Storeowner
						storeNum={this.state.storeNum}
						storeOwnerEmail={this.state.storeOwnerEmail}
					/>
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

export default StoreLoginRegister;
