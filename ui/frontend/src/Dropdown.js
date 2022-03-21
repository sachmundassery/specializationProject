import React from "react";

class Dropdown extends React.Component {
	render() {
		return (
			<div className="form-item">
				<label htmlFor={this.props.id} className="form-label">
					{this.props.label}
				</label>
				<select
					id={this.props.id}
					name={this.props.name}
					className="form-item"
					defaultValue={this.props.value}
					onChange={this.props.onChangeValue}
				>
					{this.props.valueList.map(function (value, index) {
						return (
							<option key={index} value={value}>
								{value}
							</option>
						);
					})}
				</select>
			</div>
		);
	}
}

export default Dropdown;
