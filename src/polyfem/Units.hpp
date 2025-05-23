#pragma once

#include <polyfem/Common.hpp>

#include <spdlog/fmt/ostr.h>

#include <string>

namespace polyfem
{
	class Units
	{
	public:
		void init(const json &json);

		static double convert(const json &val, const std::string &unit_type);
		static double convert(const double val, const std::string &unit, const std::string &unit_type);

		const std::string &length() const { return length_; }
		const std::string &mass() const { return mass_; }
		const std::string &time() const { return time_; }
		double characteristic_length() const { return characteristic_length_; }
		double &characteristic_length() { return characteristic_length_; }
		std::string capacitance() const { return fmt::format("{}^2/({}*{})", coulomb_, force(), length_); } // Farad
		std::string permittivity() const { return fmt::format("{}/{}", capacitance(), length_); }

		std::string stress() const { return fmt::format("{}/({}*{}^2)", mass_, length_, time_); }
		std::string density() const { return fmt::format("{}/{}^3", mass_, length_); }
		std::string velocity() const { return fmt::format("{}/{}", length_, time_); }
		std::string acceleration() const { return fmt::format("{}/{}^2", length_, time_); }
		std::string force() const { return fmt::format("{}*{}", mass_, acceleration()); }
		std::string pressure() const { return fmt::format("{}*{}/{}", mass_, acceleration(), length_); }
		std::string energy() const { return fmt::format("{}*{}^2/{}^2", mass_, length_, time_); }
		std::string viscosity() const { return fmt::format("{}/{}^2*{}", force(), length_, time_); }

	private:
		std::string length_ = "m";
		std::string mass_ = "kg";
		std::string time_ = "s";
		double characteristic_length_ = 1;
		std::string coulomb_ = "C";
	};
} // namespace polyfem
