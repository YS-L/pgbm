#ifndef DATA_H_
#define DATA_H_

#include <vector>
#include <map>

class DataMatrix {

public:
	DataMatrix();

	struct FeaturePoint {
		unsigned int sample_index;
		double value;
	};

	struct SamplePoint {
		std::map<unsigned int, double> features;
	};

	int Load(const char *filename, int skips=0, int max_num_samples=-1);
	void SetTargets(const std::vector<double>& targets);
	const SamplePoint& GetRow(unsigned int index) const;
	const std::vector<SamplePoint>& GetRows() const;
	const std::vector<FeaturePoint>& GetColumn(unsigned int index) const;
	const std::map<unsigned int, std::vector<FeaturePoint> >& GetColumns() const;
	const std::vector<double>& GetTargets() const;
	std::vector<unsigned int> GetFeatureKeys() const;

	unsigned int Size() const;
	unsigned int Dimension() const;

private:

	std::map<unsigned int, std::vector<FeaturePoint> > column_data_;
	std::vector<SamplePoint> row_data_;
	std::vector<double> targets_;

};

#endif
