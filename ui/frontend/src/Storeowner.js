import React from "react";

import Dropdown from "./Dropdown";
import MailPrediction from "./MailPrediction";
import PredictionReport from "./PredictionReport";

class Storeowner extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			storeNumber: this.props.storeNum,
			storeType: "a",
			assortment: "a",
			promo: 0,
			promo2: 0,
			promoInterval: "Jan,Apr,Jul,Oct",
			schoolHoliday: 0,
			stateHoliday: 0,
			competitionDistance: null,
			saleDate: null,
			promo2Since: null,
			competitionSince: null,
			resultPredicted: 0,
			resultStatus: false,
			printOptionEnabled: false,
			storeOwnerEmail: this.props.storeOwnerEmail,
			emailRequired: false,
		};
	}

	handleValueChange = (event) => {
		console.log(event.target.name, "-----------", event.target.value);
		if (event.target.value == "Yes") {
			this.setState({ [event.target.name]: 1 });
		} else if (event.target.value == "No") {
			this.setState({ [event.target.name]: 0 });
		} else {
			this.setState({ [event.target.name]: event.target.value });
		}
	};

	handlePrint = () => {
		this.setState({ printOptionEnabled: true });
	};
	handleEmail = () => {
		this.setState({ emailRequired: true });
		console.log("----email-----", this.state.storeOwnerEmail);
	};
	submitForm = async (event) => {
		event.preventDefault();
		const formData = new FormData();

		const compOpenSinceYear = this.state.competitionSince.split("-")[0];
		const compOpenSinceMonth = this.state.competitionSince.split("-")[1];
		const compOpenSince = compOpenSinceYear.concat("/", compOpenSinceMonth);
		formData.append("type", "predict");
		formData.append("date", this.state.saleDate);
		formData.append("promo", this.state.promo);
		formData.append("state_holiday", this.state.stateHoliday);
		formData.append("school_holiday", this.state.schoolHoliday);
		formData.append("store_type", this.state.storeType);
		formData.append("assortment", this.state.assortment);
		formData.append("competition_distance", this.state.competitionDistance);
		formData.append("competition_open_since", compOpenSince);
		formData.append("promo2", this.state.promo2);
		formData.append("promo2_since", this.state.promo2Since);
		formData.append("promo_interval", this.state.promoInterval);

		const response = await fetch("http://127.0.0.1:8000/storeowner", {
			method: "POST",
			body: formData,
		});
		const result = await response.json();
		window.alert(`Predicted Sales: ${result["prediction"]}`);
		console.log(result);
		this.setState({ resultPredicted: result["prediction"] });
		this.setState({ resultStatus: true });
	};
	render() {
		return (
			<div className="containerForm">
				<form action="" className="form-wrapper">
					<h4 style={{ marginBottom: "25px", textAlign: "center" }}>
						Store Number : {this.props.storeNum}
					</h4>

					<Dropdown
						id="store_type"
						label="Store Type"
						name="storeType"
						value={this.state.storeType}
						valueList={["a", "b", "c", "d"]}
						onChangeValue={this.handleValueChange}
					/>

					<Dropdown
						id="assortment"
						label="Assortment"
						name="assortment"
						value={this.state.assortment}
						valueList={["a", "b", "c"]}
						onChangeValue={this.handleValueChange}
					/>

					<Dropdown
						id="promo"
						label="Promo"
						name="promo"
						value={this.state.promo}
						valueList={["No", "Yes"]}
						onChangeValue={this.handleValueChange}
					/>

					<Dropdown
						id="promo_2"
						label="Promo 2"
						name="promo2"
						value={this.state.promo2}
						valueList={["No", "Yes"]}
						onChangeValue={this.handleValueChange}
					/>

					<Dropdown
						id="promo_interval"
						label="Promo Interval"
						name="promoInterval"
						value={this.state.promoInterval}
						valueList={[
							"Jan,Apr,Jul,Oct",
							"Feb,May,Aug,Nov",
							"Mar,Jun,Sept,Dec",
						]}
						onChangeValue={this.handleValueChange}
					/>

					<Dropdown
						id="school_holiday"
						label="School Holiday"
						name="schoolHoliday"
						value={this.state.schoolHoliday}
						valueList={["No", "Yes"]}
						onChangeValue={this.handleValueChange}
					/>

					<Dropdown
						id="state_holiday"
						label="State Holiday"
						name="stateHoliday"
						value={this.state.stateHoliday}
						valueList={["No", "Yes"]}
						onChangeValue={this.handleValueChange}
					/>

					<div className="form-item">
						<label htmlFor="competition_distance" className="form-label">
							Competition Distance
						</label>
						<input
							type="text"
							id="competition_distance"
							className="form-item"
							name="competitionDistance"
							onChange={this.handleValueChange}
						/>
					</div>

					<div className="form-item">
						<label htmlFor="sale_date" className="form-label">
							Sale Date
						</label>
						<input
							type="date"
							id="sale_date"
							className="form-item"
							name="saleDate"
							onChange={this.handleValueChange}
							min="2015-07-01"
						/>
					</div>

					<div className="form-item">
						<label htmlFor="promo2_since" className="form-label">
							Promo 2 Start Date
						</label>
						<input
							type="date"
							id="promo2_since"
							className="form-item"
							name="promo2Since"
							onChange={this.handleValueChange}
							min="2005-01-01"
							max="2012-12-31"
						/>
					</div>

					<div className="form-item">
						<label htmlFor="competition_since" className="form-label">
							Competition Start Date
						</label>
						<input
							type="date"
							id="competition_since"
							className="form-item"
							name="competitionSince"
							onChange={this.handleValueChange}
							min="2005-01-01"
							max="2012-12-31"
						/>
					</div>

					<button className="printAndSend" onClick={this.submitForm}>
						Predict
					</button>
				</form>
				{this.state.resultStatus && (
					<div className="btnDiv">
						<button onClick={this.handlePrint} className="printAndSend">
							Print
						</button>
					</div>
				)}

				{this.state.resultStatus && (
					<>
						<MailPrediction
							predictedResult={this.state.resultPredicted}
							storeNum={this.props.storeNum}
							emailRequired={this.props.storeOwnerEmail}
						/>
					</>
				)}

				{this.state.printOptionEnabled && (
					<PredictionReport
						predictedResult={this.state.resultPredicted}
						storeNum={this.props.storeNum}
					/>
				)}
			</div>
		);
	}
}

export default Storeowner;
