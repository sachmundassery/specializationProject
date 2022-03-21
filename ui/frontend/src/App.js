import React from "react";
import About from "./About";
import ContactUs from "./ContactUs";
import Home from "./Home";
import NavigationBar from "./NavigationBar";
import Team from "./Team";
// import { Route, Link } from "react-router-dom";
// import Storeowner from "./Storeowner";
// import Report from "./Report";

function App() {
	return (
		<>
			<NavigationBar />
			{/* <Route path="/report" component={Report} /> */}
			<Home />

			<About />

			<Team />
			<ContactUs />
		</>
	);
}

export default App;
